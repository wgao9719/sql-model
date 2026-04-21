"""
AT&T-style initial-state builder for BIRD text-to-SQL.

Two stages:

    build-db-meta   one-shot per DB. Walks every (table, column), runs cheap
                    sqlite introspection (count, distinct-count, null fraction,
                    min/max for numeric, 3 example values) and attaches BIRD's
                    column descriptions where present. Output: JSON blob
                    (db_id → per-column metadata), ~1-5 MB for all 69 train DBs.

    build-states    consumes the above + a split JSON (split_train / split_val /
                    mini_dev_sqlite), assembles the enriched `initial_state`
                    per question, writes JSONL.

The `initial_state` string is the single artifact used for SFT prompts, RL
rollouts, and Mini-Dev inference — same format everywhere.

CLI:
    python -m finetuning.rl.metadata_builder build-db-meta \
        --tables         finetuning/train_data/train_tables.json \
        --db-root        finetuning/train_data/train_databases \
        --column-meaning finetuning/train_data/train_column_meaning.json \
        --out            finetuning/train_data/db_metadata.json

    python -m finetuning.rl.metadata_builder build-states \
        --input   finetuning/train_data/split_train.json \
        --db-meta finetuning/train_data/db_metadata.json \
        --out     finetuning/train_data/split_train_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

from tqdm import tqdm

SAMPLE_VALUE_COUNT = 3
SAMPLE_VALUE_MAX_LEN = 40
QUERY_TIMEOUT_SEC = 10
NDISTINCT_ROW_LIMIT = 1_000_000
NUMERIC_TYPES = {"integer", "int", "real", "numeric", "number", "float", "double"}


# ----------------------------------------------------------------------------
# sqlite helpers
# ----------------------------------------------------------------------------

def _dec(x: Any) -> Any:
    # BIRD DBs occasionally have mixed-encoding bytes in text columns; decode defensively.
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.decode("latin-1", errors="replace")
    return x


def _exec(conn: sqlite3.Connection, sql: str, timeout: int = QUERY_TIMEOUT_SEC) -> list | None:
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return [tuple(_dec(v) for v in row) for row in rows]
    except sqlite3.Error:
        return None
    finally:
        timer.cancel()


def _safe_str(v: Any) -> str:
    s = str(v) if v is not None else "NULL"
    if len(s) > SAMPLE_VALUE_MAX_LEN:
        s = s[:SAMPLE_VALUE_MAX_LEN] + "..."
    return s


def _column_stats(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> dict:
    t = f"`{table}`"
    c = f"`{column}`"
    out: dict = {}

    n = _exec(conn, f"SELECT COUNT(*) FROM {t}")
    n_rows = (n[0][0] or 0) if n else 0
    out["n_rows"] = n_rows
    if n_rows == 0:
        return out

    nulls = _exec(conn, f"SELECT SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM {t}")
    null_count = (nulls[0][0] or 0) if nulls else 0
    out["null_frac"] = round(null_count / n_rows, 3)

    # COUNT(DISTINCT) is the expensive one — skip on huge tables.
    if n_rows <= NDISTINCT_ROW_LIMIT:
        ndist = _exec(conn, f"SELECT COUNT(DISTINCT {c}) FROM {t}")
        if ndist is not None:
            out["n_distinct"] = ndist[0][0]

    if col_type.lower() in NUMERIC_TYPES:
        rng = _exec(conn, f"SELECT MIN({c}), MAX({c}) FROM {t} WHERE {c} IS NOT NULL")
        if rng and rng[0][0] is not None:
            out["min"], out["max"] = rng[0]

    samples = _exec(conn, f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT {SAMPLE_VALUE_COUNT}")
    if samples:
        out["examples"] = [_safe_str(row[0]) for row in samples]
    return out


# ----------------------------------------------------------------------------
# DB metadata builder
# ----------------------------------------------------------------------------

def _render_fk(col_idx: int, fk_map: dict, col_names: list, table_names: list) -> str | None:
    dst_idx = fk_map.get(col_idx)
    if dst_idx is None:
        return None
    dst_table_idx, dst_col_name = col_names[dst_idx]
    return f"{table_names[dst_table_idx]}.{dst_col_name}"


def build_db_metadata(db_id: str, db_path: str, tables_entry: dict, column_meaning: dict) -> dict:
    conn = sqlite3.connect(db_path)
    conn.text_factory = bytes
    try:
        tn = tables_entry["table_names_original"]
        col_names = tables_entry["column_names_original"]
        col_types = tables_entry["column_types"]
        pk_idxs = set(tables_entry.get("primary_keys", []))
        fk_map = {src: dst for src, dst in tables_entry.get("foreign_keys", [])}

        tables: list[dict] = []
        for t_idx, t_name in enumerate(tn):
            cols_info = []
            for c_idx, (t_ref, c_name) in enumerate(col_names):
                if t_ref != t_idx:
                    continue
                c_type = col_types[c_idx]
                try:
                    stats = _column_stats(conn, t_name, c_name, c_type)
                except Exception as e:
                    stats = {"error": str(e)}
                hint_key = f"{db_id}|{t_name.lower()}|{c_name.lower()}"
                cols_info.append({
                    "idx": c_idx,
                    "name": c_name,
                    "type": c_type,
                    "pk": c_idx in pk_idxs,
                    "fk_to": _render_fk(c_idx, fk_map, col_names, tn),
                    "hint": column_meaning.get(hint_key),
                    "stats": stats,
                })
            tables.append({"name": t_name, "columns": cols_info})
        return {"db_id": db_id, "tables": tables}
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------------

TASK_PREAMBLE = (
    "You are a data-science expert. Given a SQLite database schema with rich column-level "
    "metadata and a natural-language question, produce one valid SQLite query that answers "
    "the question. Return only the SQL, wrapped in a ```sql fenced block."
)


def _render_column(c: dict) -> list[str]:
    stats = c.get("stats", {}) or {}
    parts = [c["name"], c["type"]]
    if c["pk"]:
        parts.append("PK")
    if c.get("fk_to"):
        parts.append(f"FK->{c['fk_to']}")
    summary = []
    if "n_distinct" in stats:
        summary.append(f"n_distinct={stats['n_distinct']}")
    if stats.get("null_frac", 0) > 0:
        summary.append(f"null_frac={stats['null_frac']}")
    if "min" in stats and "max" in stats:
        summary.append(f"range=[{stats['min']}, {stats['max']}]")
    if "examples" in stats:
        summary.append("ex=" + ", ".join(repr(x) for x in stats["examples"]))
    line = "  " + " ".join(parts)
    if summary:
        line += "  -- " + "; ".join(summary)
    lines = [line]
    hint = (c.get("hint") or "").replace("\n", " ").strip()
    if hint:
        if len(hint) > 200:
            hint = hint[:200] + "..."
        lines.append(f"    # {hint}")
    return lines


def _render_table(t: dict) -> str:
    out = [f"TABLE {t['name']}"]
    for c in t["columns"]:
        out.extend(_render_column(c))
    return "\n".join(out)


def build_initial_state(question: str, evidence: str | None, db_id: str, db_meta: dict) -> str:
    assert db_meta["db_id"] == db_id, f"db_meta {db_meta['db_id']} != requested {db_id}"
    schema = "\n\n".join(_render_table(t) for t in db_meta["tables"])
    evidence = (evidence or "").strip()
    q_block = f"[QUESTION]\n{question.strip()}"
    if evidence:
        q_block += f"\n\n[EVIDENCE]\n{evidence}"
    return (
        f"[TASK]\n{TASK_PREAMBLE}\n\n"
        f"{q_block}\n\n"
        f"[DATABASE] {db_id}\n"
        f"[SCHEMA]\n{schema}\n\n"
        f"[SQL]\n"
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _cmd_build_db_meta(args: argparse.Namespace) -> None:
    tables = json.load(open(args.tables))
    column_meaning = json.load(open(args.column_meaning)) if args.column_meaning else {}
    out: dict = {}
    for entry in tqdm(tables, desc="dbs"):
        db_id = entry["db_id"]
        db_path = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            print(f"skip missing {db_path}", file=sys.stderr)
            continue
        out[db_id] = build_db_metadata(db_id, str(db_path), entry, column_meaning)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} DB metadata records to {args.out}")


def _cmd_build_states(args: argparse.Namespace) -> None:
    rows = json.load(open(args.input))
    db_meta = json.load(open(args.db_meta))
    lines: list[str] = []
    missing: set = set()
    for r in tqdm(rows, desc="rows"):
        db_id = r["db_id"]
        if db_id not in db_meta:
            missing.add(db_id)
            continue
        state = build_initial_state(
            question=r["question"],
            evidence=r.get("evidence"),
            db_id=db_id,
            db_meta=db_meta[db_id],
        )
        lines.append(json.dumps({
            "initial_state": state,
            "db_id": db_id,
            "gold_sql": r.get("SQL", r.get("gt_sql", "")),
            "question": r["question"],
        }))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} enriched states to {args.out}")
    if missing:
        dropped = sum(1 for r in rows if r["db_id"] in missing)
        print(f"warning: skipped {dropped} rows from unknown DBs: {sorted(missing)[:5]}...",
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("build-db-meta")
    a.add_argument("--tables", required=True)
    a.add_argument("--db-root", required=True)
    a.add_argument("--column-meaning", default=None)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=_cmd_build_db_meta)

    b = sub.add_parser("build-states")
    b.add_argument("--input", required=True)
    b.add_argument("--db-meta", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=_cmd_build_states)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
