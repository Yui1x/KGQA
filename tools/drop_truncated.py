#!/usr/bin/env python
"""
Remove records whose answer was lost to the output-token cap, so that a resumed
run regenerates only those questions with a larger cap.

Why this is equivalent to rerunning everything with the larger cap: max_tokens
is only a ceiling. An output that finished below the old cap would come out the
same under a higher one (temperature 0), so only the capped outputs can change.

A record is dropped when
  * usage.finish_reasons has "length" (runs made after finish reasons were logged), or
  * the record has no finish reasons (older runs) and its answer is exactly one
    of the runner's parse-failure fallbacks with empty evidence.

The original file is kept as <split>.jsonl.bak_<timestamp>; dropped records are
also written to <split>.dropped_<timestamp>.jsonl.

Usage:
    python tools/drop_truncated.py output/bl-ds/zero_shot output/bl-ds/few_shot output/bl-ds/cot
    python tools/drop_truncated.py --dry-run output/base-ds/tog output/base-ds/tog2 output/base-ds/search_on_graph
then rerun the same tag with a larger cap, e.g.
    python tools/run_baselines.py --tag bl-ds --methods zero_shot,few_shot,cot --max-length 8192 ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

FALLBACK_ANSWERS = {"无法回答。", "证据不足，无法回答。"}
SPLITS = ("single_hop", "multi_hop", "hallucination")


def should_drop(rec: dict) -> bool:
    usage = rec.get("usage") or {}
    reasons = usage.get("finish_reasons")
    if isinstance(reasons, dict):
        return int(reasons.get("length", 0) or 0) > 0
    return (str(rec.get("answer") or "").strip() in FALLBACK_ANSWERS
            and not str(rec.get("evidence") or "").strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="method directories containing <split>.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be dropped")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = 0
    for d in args.dirs:
        for split in SPLITS:
            path = Path(d) / f"{split}.jsonl"
            if not path.is_file():
                continue
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            keep, drop = [], []
            for ln in lines:
                (drop if should_drop(json.loads(ln)) else keep).append(ln)
            qids = sorted(int(json.loads(ln)["qid"]) for ln in drop)
            print(f"{path}: {len(lines)} records, drop {len(drop)} {qids if qids else ''}")
            total += len(drop)
            if drop and not args.dry_run:
                shutil.copy2(path, path.with_name(f"{split}.jsonl.bak_{stamp}"))
                path.with_name(f"{split}.dropped_{stamp}.jsonl").write_text("\n".join(drop) + "\n", encoding="utf-8")
                path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"{'would drop' if args.dry_run else 'dropped'} {total} records in total")


if __name__ == "__main__":
    main()
