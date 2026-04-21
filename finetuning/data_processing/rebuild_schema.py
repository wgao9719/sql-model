#!/usr/bin/env python3
"""Inject profile-generated column descriptions into training and inference prompts."""

import argparse
import json
import os
import re
from glob import glob


# Matches a column line: `  name type, -- [desc,] example: [...]` — replaces desc only.
_COL_LINE_RE = re.compile(
    r"""
    ^(?P<indent>\s*)
    (?P<name>[`"\w]+)\s+
    (?P<dtype>[^,\n]+?)
    ,\s*--\s*
    (?P<before_ex>.*?)
    (?P<example>example:\s*\[.*?\])
    \s*$
    """,
    re.VERBOSE,
)

# Matches a full CREATE TABLE block — captures name and body.
_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+`?\"?(?P<tname>[\w]+)\"?`?\s*\((?P<body>[\s\S]*?)\)\s*;",
    re.IGNORECASE,
)


def _strip_bird_description(before_example: str) -> str:
    """Drop any existing BIRD description preceding `example:` — we re-inject our own."""
    return ""


def _compose_col_comment(new_desc: str, example_block: str) -> str:
    """Compose the full '-- [new_desc,] example: [...]' comment tail."""
    if new_desc:
        return f"-- {new_desc}, {example_block}"
    return f"-- {example_block}"


def rewrite_schema_block(schema_text: str, descriptions_for_db: dict) -> str:
    """Rewrite every column line's description inside the schema."""
    def _rewrite_table(m: re.Match) -> str:
        tname = m.group("tname")
        body = m.group("body")
        col_descs = descriptions_for_db.get(tname, {})
        new_lines = []
        # keepends=True preserves trailing newlines so rejoin round-trips exactly when unchanged.
        for raw_line in body.splitlines(keepends=True):
            # Strip newline for regex match, re-append after.
            trailing = "\n" if raw_line.endswith("\n") else ""
            line = raw_line[:-1] if trailing else raw_line
            m2 = _COL_LINE_RE.match(line)
            if not m2:
                new_lines.append(raw_line)
                continue
            col_name = m2.group("name").strip('`"')
            new_desc = col_descs.get(col_name, "")
            if not new_desc:
                new_lines.append(raw_line)
                continue
            new_comment = _compose_col_comment(new_desc, m2.group("example"))
            new_line = f"{m2.group('indent')}{m2.group('name')} {m2.group('dtype')}, {new_comment}{trailing}"
            new_lines.append(new_line)
        new_body = "".join(new_lines)
        return f"CREATE TABLE {tname} ({new_body});"

    return _TABLE_RE.sub(_rewrite_table, schema_text)


def splice_prompt(original_prompt: str, new_schema_block: str) -> str:
    """Swap the `Database Schema:\n...\n<post>` block for new_schema_block in place."""
    pre = "Database Schema:\n"
    post_marker = "\nThis schema describes the database"
    i = original_prompt.find(pre)
    j = original_prompt.find(post_marker)
    if i == -1 or j == -1 or j <= i:
        # Minidev prompts lack the boilerplate; fall back to the first `\nQuestion:` marker.
        j = original_prompt.find("\nQuestion:")
        if i == -1 or j == -1 or j <= i:
            return original_prompt
    before = original_prompt[: i + len(pre)]
    after = original_prompt[j:]
    return before + new_schema_block.rstrip() + after


def load_descriptions_dir(path: str) -> dict:
    """Load descriptions/{db_id}.json into {db_id: {table: {col: desc}}}."""
    out = {}
    for p in sorted(glob(os.path.join(path, "*.json"))):
        db_id = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            out[db_id] = json.load(f)
    return out


def transform_train_raw(in_path: str, out_path: str, descriptions: dict) -> tuple:
    """Rewrite db_desc + input_seq in train_bird_raw.json using db_id from split_train.json."""
    # db_id isn't in raw — pull it from split_train.json alongside.
    split_path = os.path.join(os.path.dirname(in_path), "split_train.json")
    with open(split_path) as f:
        split = json.load(f)
    assert len(split), "empty split_train.json"
    with open(in_path) as f:
        raw = json.load(f)
    assert len(raw) == len(split), f"row count mismatch: raw={len(raw)} split={len(split)}"

    changed = 0
    for i, row in enumerate(raw):
        db_id = split[i]["db_id"]
        d = descriptions.get(db_id)
        if not d:
            continue
        new_schema = rewrite_schema_block(row["db_desc"], d)
        if new_schema != row["db_desc"]:
            row["db_desc"] = new_schema
            row["input_seq"] = splice_prompt(row["input_seq"], new_schema)
            changed += 1
    with open(out_path, "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return changed, len(raw)


def transform_minidev(in_path: str, out_path: str, descriptions: dict) -> tuple:
    """Rewrite schema + prompt in mini_dev_prompt.jsonl (line-delimited JSON)."""
    changed = total = 0
    with open(in_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            row = json.loads(line)
            total += 1
            d = descriptions.get(row.get("db_id"))
            if d:
                new_schema = rewrite_schema_block(row.get("schema", ""), d)
                if new_schema != row.get("schema"):
                    row["schema"] = new_schema
                    row["prompt"] = splice_prompt(row["prompt"], new_schema)
                    changed += 1
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed, total


def main():
    ap = argparse.ArgumentParser("Inject profile descriptions into prompts")
    ap.add_argument("--descriptions", required=True)
    ap.add_argument("--train-raw-in", default="")
    ap.add_argument("--train-raw-out", default="")
    ap.add_argument("--minidev-in", default="")
    ap.add_argument("--minidev-out", default="")
    args = ap.parse_args()

    descriptions = load_descriptions_dir(args.descriptions)
    print(f"[rebuild_schema] loaded descriptions for {len(descriptions)} DBs")

    if args.train_raw_in and args.train_raw_out:
        changed, total = transform_train_raw(args.train_raw_in, args.train_raw_out, descriptions)
        print(f"[rebuild_schema] train: changed {changed}/{total} rows -> {args.train_raw_out}")

    if args.minidev_in and args.minidev_out:
        changed, total = transform_minidev(args.minidev_in, args.minidev_out, descriptions)
        print(f"[rebuild_schema] minidev: changed {changed}/{total} rows -> {args.minidev_out}")


if __name__ == "__main__":
    main()
