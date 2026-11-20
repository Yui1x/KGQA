#!/usr/bin/env python
"""
E6: run the system's own joint sufficiency judgement on the pairs built by
tools/e6_build_pairs.py, once per pair.

The call is search._judge_joint_sufficiency with the frozen settings of the
full system (joint mode, structural rule on, judge_max_tokens 4096, temperature
0), so the decision includes the hop-count precheck and the deterministic
validation of the support indices, exactly as in a real run. No Neo4j needed.

Resumable: pairs already present in --out are skipped.

Usage (Windows, same gateway as the main runs):
    python tools/e6_run_judge.py --pairs output/e6/pairs.jsonl --out output/e6/verdicts.jsonl ^
        --model cn:deepseek-v4.1-flash --api-base http://127.0.0.1:7864/v1 --api-key ... --concurrency 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "KG_QA"))

import search  # noqa: E402
from utils import configure_client, configure_http_pool, render_path_chain, track_llm_usage  # noqa: E402


def build_args(ns: argparse.Namespace) -> SimpleNamespace:
    """Only the attributes the judgement reads; values identical to the full system."""
    return SimpleNamespace(
        LLM_type=ns.model,
        api_base=ns.api_base,
        opeani_api_keys=ns.api_key[0] if ns.api_key else "",
        temperature_reasoning=0.0,
        max_length=ns.max_length,
        judge_max_tokens=ns.judge_max_tokens,
        top_k_paths_for_reasoning=5,
        ablation="full",
        sufficiency_mode="joint",
        structural_rule="on",
    )


def judge_one(pair: dict, args: SimpleNamespace) -> dict:
    min_hops = search.required_hops(args, pair["slots"])
    started = time.time()
    with track_llm_usage() as usage:
        verdict = search._judge_joint_sufficiency(
            question=pair["question"], entity_name=pair["entity"], pool=pair["pool"], clue="",
            args=args, query_slots=pair["slots"], min_hops=min_hops,
        )
    return {
        "pid": pair["pid"],
        "sufficient": bool(verdict["sufficient"]),
        "support": [i + 1 for i in verdict.get("support", [])],
        "support_paths": [render_path_chain(pair["pool"][i]["triplets"]) for i in verdict.get("support", [])],
        "slot_support": verdict.get("slot_support", {}),
        "missing": verdict.get("missing", ""),
        "reject_reason": verdict.get("reject_reason", ""),
        "llm_called": bool(verdict.get("llm_called")),
        "min_hops": min_hops,
        "model": args.LLM_type,
        "usage": usage.to_dict(),
        "seconds": round(time.time() - started, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the joint sufficiency judgement on E6 pairs.")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("KGQA_MODEL", "cn:deepseek-v4.1-flash"))
    ap.add_argument("--api-base", default=os.getenv("KGQA_API_BASE", ""))
    ap.add_argument("--api-key", action="append", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N pairs")
    ns = ap.parse_args()
    if not ns.api_key:
        env_key = os.getenv("KGQA_API_KEY", "")
        ns.api_key = [env_key] if env_key else []
    if not ns.api_key or not ns.api_base:
        sys.exit("need --api-base and --api-key (without a key the judge would silently accept every pool)")

    configure_client(ns.api_key, 0)
    configure_http_pool(ns.concurrency + 8)
    args = build_args(ns)
    pairs = [json.loads(l) for l in Path(ns.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        done = {json.loads(l)["pid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [p for p in pairs if p["pid"] not in done]
    if ns.limit:
        todo = todo[: ns.limit]
    print(f"{len(pairs)} pairs, {len(done)} done, {len(todo)} to judge with {args.LLM_type}")
    lock = threading.Lock()
    finished = 0
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=ns.concurrency) as pool:
        futures = {pool.submit(judge_one, p, args): p["pid"] for p in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - log and continue; rerun picks it up
                print(f"  {futures[fut]} failed: {exc}")
                continue
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                finished += 1
                if finished % 25 == 0 or finished == len(todo):
                    print(f"  {finished}/{len(todo)}")
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    trunc = sum(int((r.get("usage") or {}).get("finish_reasons", {}).get("length", 0) or 0) for r in rows)
    print(f"wrote {out}: {len(rows)} verdicts, {sum(r['sufficient'] for r in rows)} sufficient, "
          f"{sum(r['llm_called'] for r in rows)} with an LLM call, {trunc} truncated calls")


if __name__ == "__main__":
    main()
