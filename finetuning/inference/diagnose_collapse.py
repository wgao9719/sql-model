"""Quantify SFT output-distribution collapse — compares candidate diversity across models."""
import argparse
import json
import os
import statistics as s
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_orch import _execute_sqlite, _db_path, _result_hash
from modal_infer_orch import _extract_sql


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pairwise_edit(strs):
    if len(strs) < 2:
        return 0.0
    total, n = 0.0, 0
    for i in range(len(strs)):
        for j in range(i + 1, len(strs)):
            total += 1.0 - SequenceMatcher(None, strs[i], strs[j]).ratio()
            n += 1
    return total / n


def _exec_wrap(args):
    sql, db_path, timeout = args
    return _execute_sqlite(sql, db_path, timeout)


def audit(name, raw_path, db_root, exec_timeout=5.0, workers=8):
    rows = load_jsonl(raw_path)
    n_q = len(rows)

    distinct_strs, distinct_sqls, pair_eds, lens_all = [], [], [], []
    first_tok = Counter()

    exec_jobs = []
    row_slice = []
    for r in rows:
        cands = r["candidates"]
        sqls = [_extract_sql(c) or "" for c in cands]
        distinct_strs.append(len(set(cands)))
        distinct_sqls.append(len(set(sqls)))
        pair_eds.append(pairwise_edit(cands))
        lens_all.extend(len(c) for c in cands)
        for sql in sqls:
            first = sql.strip().split()[0] if sql.strip() else "<empty>"
            first_tok[first.upper()] += 1
        db = _db_path(db_root, r["db_id"], "SQLite")
        start = len(exec_jobs)
        for sql in sqls:
            exec_jobs.append((sql, db, exec_timeout))
        row_slice.append((start, start + len(sqls)))

    print(f"  [{name}] executing {len(exec_jobs)} candidates with {workers} workers...",
          flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        all_results = list(pool.map(_exec_wrap, exec_jobs, chunksize=16))

    distinct_hashes = []
    for (s_i, e_i) in row_slice:
        hashes = set()
        for res in all_results[s_i:e_i]:
            if res["ok"] and res["rows"]:
                hashes.add(_result_hash(res["rows"]))
        distinct_hashes.append(len(hashes))

    def summarize(label, xs):
        return f"{label}: mean={s.mean(xs):.2f}  median={s.median(xs):.1f}  p95={sorted(xs)[int(0.95*len(xs))]:.0f}"

    total_samples = sum(len(r["candidates"]) for r in rows)
    print(f"\n=== {name}  ({n_q} questions, {total_samples} candidates) ===")
    print(f"  1. {summarize('distinct response strings / q', distinct_strs)}")
    print(f"  2. {summarize('distinct extracted SQLs  / q', distinct_sqls)}")
    print(f"  3. {summarize('distinct result hashes   / q', distinct_hashes)}")
    print(f"  4. mean pairwise edit dist             {s.mean(pair_eds):.3f}")
    print(f"  5. response length (chars)   mean={s.mean(lens_all):.0f}  stdev={s.stdev(lens_all):.0f}")
    print(f"  6. first-token top-5 across all samples  {first_tok.most_common(5)}")
    return {
        "name": name,
        "n_q": n_q,
        "distinct_strs_mean": s.mean(distinct_strs),
        "distinct_sqls_mean": s.mean(distinct_sqls),
        "distinct_hashes_mean": s.mean(distinct_hashes),
        "pair_edit_mean": s.mean(pair_eds),
        "resp_len_mean": s.mean(lens_all),
        "resp_len_stdev": s.stdev(lens_all),
        "first_tok_top5": first_tok.most_common(5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--exec-timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("raw", nargs="+",
                    help="name=path pairs, e.g. base=/abs/base.jsonl sft=/abs/sft.jsonl")
    args = ap.parse_args()
    pairs = [r.split("=", 1) for r in args.raw]
    results = []
    for name, path in pairs:
        results.append(audit(name, path, args.db_root, args.exec_timeout, args.workers))

    print("\n=== SUMMARY (lower = more collapsed) ===")
    print(f"{'model':>20}  {'distinct_sql/q':>14}  {'distinct_hash/q':>14}  {'edit_dist':>10}  {'len_mean':>9}  {'len_std':>9}")
    for r in results:
        print(f"{r['name']:>20}  {r['distinct_sqls_mean']:>14.2f}  {r['distinct_hashes_mean']:>14.2f}  "
              f"{r['pair_edit_mean']:>10.3f}  {r['resp_len_mean']:>9.0f}  {r['resp_len_stdev']:>9.0f}")


if __name__ == "__main__":
    main()
