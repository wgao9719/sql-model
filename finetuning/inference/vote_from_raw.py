"""Local SC voter: load raw candidates, execute with timeout, write separator-correct JSON."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_orch import _sc_vote, _db_path, BIRD_SEP
from modal_infer_orch import _extract_sql


def _worker(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference_orch import _sc_vote
    extracted, db_path, timeout = args
    return _sc_vote(extracted, db_path, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    args = ap.parse_args()

    db_root_abs = os.path.abspath(args.db_root)
    rows = []
    with open(args.raw) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[vote] {len(rows)} questions", flush=True)

    jobs = []
    for r in rows:
        extracted = [_extract_sql(c) or "" for c in r["candidates"]]
        db_path = _db_path(db_root_abs, r["db_id"], "SQLite")
        jobs.append((extracted, db_path, args.exec_timeout))

    print(f"[vote] voting with {args.workers} workers", flush=True)
    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        final_sqls = []
        for i, sql in enumerate(pool.map(_worker, jobs, chunksize=4)):
            final_sqls.append(sql)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                dt = time.time() - t0
                print(f"[vote] {i+1}/{len(jobs)}  ({dt:.1f}s elapsed)", flush=True)

    payload = {
        str(i): f"{sql}{BIRD_SEP}{r['db_id']}"
        for i, (sql, r) in enumerate(zip(final_sqls, rows))
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[vote] wrote {len(final_sqls)} predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
