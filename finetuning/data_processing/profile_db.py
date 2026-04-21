#!/usr/bin/env python3
"""Profile every column of every SQLite DB to profiles/{db_id}.json."""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from glob import glob

SAMPLE_LIMIT = 20000
TOP_K = 10


_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_URL = re.compile(r"^https?://")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _detect_pattern(sample):
    """Return a coarse pattern label if ≥80% of non-null string samples match."""
    strs = [str(v) for v in sample if v is not None]
    if not strs:
        return None
    thresh = max(1, int(0.8 * len(strs)))
    # ISO-currency intentionally omitted: 3-letter-uppercase matches country/segment codes too.
    for pat, label in [
        (_UUID, "uuid"),
        (_URL, "url"),
        (_EMAIL, "email"),
        (_DATE, "date"),
    ]:
        if sum(bool(pat.match(s)) for s in strs) >= thresh:
            return label
    return None


def _cardinality_class(distinct, non_null):
    if non_null == 0:
        return "empty"
    if distinct <= 20:
        return "enum"
    if distinct / non_null < 0.05:
        return "low"
    if distinct / non_null > 0.95:
        return "high"
    return "medium"


def _is_numeric(dtype: str) -> bool:
    d = (dtype or "").upper()
    return any(k in d for k in ["INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"])


def profile_column(conn, table, col, dtype, total_rows):
    q = f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {SAMPLE_LIMIT}'
    try:
        rows = [r[0] for r in conn.execute(q).fetchall()]
    except sqlite3.Error:
        return {"error": "could not read"}
    non_null = len(rows)
    try:
        distinct = conn.execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"'
        ).fetchone()[0]
    except sqlite3.Error:
        distinct = len(set(rows))

    top = Counter(rows).most_common(TOP_K)
    sample = rows[:200]
    pattern = _detect_pattern(sample)

    mn = mx = None
    if _is_numeric(dtype) and rows:
        try:
            mn, mx = min(rows), max(rows)
        except TypeError:
            mn = mx = None
    elif pattern == "date" and rows:
        str_rows = [str(v) for v in rows if v is not None]
        if str_rows:
            mn, mx = min(str_rows), max(str_rows)

    return {
        "dtype": dtype,
        "total_rows": total_rows,
        "non_null_count": non_null,
        "distinct": int(distinct or 0),
        "null_frac": round(1 - (non_null / max(total_rows, 1)), 4),
        "top_values": [[v, c] for v, c in top],
        "min": mn,
        "max": mx,
        "pattern": pattern,
        "cardinality_class": _cardinality_class(distinct, non_null),
    }


def profile_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]

    # PK/FK maps: table -> set(pk_cols) / {from_col: "to_table.to_col"}
    pk_map = {}
    fk_map = {}
    for t in tables:
        pk_map[t] = set()
        fk_map[t] = {}
        for row in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
            if row[5]:
                pk_map[t].add(row[1])
        for row in conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
            fk_map[t][row[3]] = f"{row[2]}.{row[4]}"

    out = {}
    for t in tables:
        try:
            total = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error:
            continue
        cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        out[t] = {}
        for row in cols:
            col_name = row[1]
            col_type = row[2]
            prof = profile_column(conn, t, col_name, col_type, total)
            prof["is_pk"] = col_name in pk_map[t]
            prof["is_fk"] = col_name in fk_map[t]
            prof["fk_target"] = fk_map[t].get(col_name)
            out[t][col_name] = prof
    conn.close()
    return out


def find_sqlite(db_dir: str):
    """First .sqlite/.db file inside db_dir — BIRD ships one per DB."""
    cands = sorted(glob(os.path.join(db_dir, "*.sqlite"))) + \
            sorted(glob(os.path.join(db_dir, "*.sqlite3"))) + \
            sorted(glob(os.path.join(db_dir, "*.db")))
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser("DB profiler")
    ap.add_argument("--db-root", required=True,
                    help="Dir containing one subdir per db_id with a .sqlite inside")
    ap.add_argument("--out", required=True, help="Output dir for {db_id}.json files")
    ap.add_argument("--overwrite", action="store_true", help="Re-profile even if output exists")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db_dirs = sorted(d for d in glob(os.path.join(args.db_root, "*")) if os.path.isdir(d))
    print(f"[profile_db] found {len(db_dirs)} candidate DB dirs under {args.db_root}")

    done = skipped = errored = 0
    for db_dir in db_dirs:
        db_id = os.path.basename(db_dir)
        out_path = os.path.join(args.out, f"{db_id}.json")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        sqlite_path = find_sqlite(db_dir)
        if not sqlite_path:
            print(f"[profile_db] SKIP {db_id}: no .sqlite file in {db_dir}", file=sys.stderr)
            errored += 1
            continue
        try:
            prof = profile_db(sqlite_path)
        except Exception as e:
            print(f"[profile_db] ERROR {db_id}: {type(e).__name__}: {e}", file=sys.stderr)
            errored += 1
            continue
        with open(out_path, "w") as f:
            json.dump(prof, f, indent=2, default=str)
        n_cols = sum(len(t) for t in prof.values())
        print(f"[profile_db] {db_id}: {len(prof)} tables, {n_cols} columns -> {out_path}")
        done += 1

    print(f"[profile_db] done={done} skipped={skipped} errored={errored}")


if __name__ == "__main__":
    main()
