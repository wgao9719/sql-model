#!/usr/bin/env python3
"""Template-mode: turn column profiles into one-line descriptions per db_id."""

import argparse
import json
import os
from glob import glob


_PATTERN_LABEL = {
    "uuid": "UUID identifier",
    "url": "URL",
    "email": "email address",
    "iso-currency": "ISO 4217 currency code",
    "date": "date",
}


def _fmt_values(top_values, k=5):
    vs = [repr(v)[:30] for v, _ in top_values[:k]]
    return ", ".join(vs)


def describe(col_name: str, p: dict) -> str:
    """Collapse a column profile into a one-line description."""
    if p.get("error"):
        return ""

    bits = []

    # Pattern hint first — most informative when present.
    pat = p.get("pattern")
    if pat and pat in _PATTERN_LABEL:
        bits.append(_PATTERN_LABEL[pat])

    # Cardinality class / distinct count.
    cls = p.get("cardinality_class")
    distinct = p.get("distinct", 0)
    if cls == "enum" and p.get("top_values"):
        vals = _fmt_values(p["top_values"], k=5)
        bits.append(f"enum of {distinct} ({vals})")
    elif cls == "high":
        bits.append(f"{distinct} distinct (high cardinality)")
    elif cls == "low":
        bits.append(f"{distinct} distinct (low cardinality)")
    elif distinct > 0:
        bits.append(f"{distinct} distinct")

    # Range for numerics/dates.
    mn, mx = p.get("min"), p.get("max")
    if mn is not None and mx is not None and mn != mx:
        bits.append(f"range [{mn}..{mx}]")

    # Null frequency — only call out if meaningful.
    null_frac = p.get("null_frac", 0) or 0
    if null_frac == 0 and p.get("total_rows", 0) > 0:
        bits.append("no nulls")
    elif null_frac >= 0.1:
        bits.append(f"{null_frac*100:.0f}% null")

    # Relationship flags.
    if p.get("is_pk"):
        bits.append("primary key")
    if p.get("is_fk") and p.get("fk_target"):
        bits.append(f"FK -> {p['fk_target']}")

    return "; ".join(bits)


def describe_db(profile: dict) -> dict:
    out = {}
    for table, cols in profile.items():
        out[table] = {}
        for col, p in cols.items():
            out[table][col] = describe(col, p)
    return out


def main():
    ap = argparse.ArgumentParser("Template-mode column descriptor")
    ap.add_argument("--profiles", required=True, help="Dir with {db_id}.json from profile_db.py")
    ap.add_argument("--out", required=True, help="Output dir for {db_id}.json descriptions")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob(os.path.join(args.profiles, "*.json")))
    print(f"[describe_columns] {len(files)} profiles under {args.profiles}")

    for path in files:
        db_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            prof = json.load(f)
        desc = describe_db(prof)
        out_path = os.path.join(args.out, f"{db_id}.json")
        with open(out_path, "w") as f:
            json.dump(desc, f, indent=2, ensure_ascii=False)
        n_cols = sum(len(t) for t in desc.values())
        print(f"[describe_columns] {db_id}: {n_cols} descriptions -> {out_path}")


if __name__ == "__main__":
    main()
