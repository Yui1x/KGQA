#!/usr/bin/env python
"""
Concurrent, resumable runner for SCEPS evaluation splits.

The stock KG_QA/main.py walks questions one at a time, which makes a 700-question
sweep an 8-hour proposition. Graph queries and LLM calls are independent per
question, so this runner fans them out over a thread pool while keeping the
exact same per-question pipeline:

    extract_query_slots -> extract_topic_entities -> search_single_entity (per
    entity) -> finalize_entity_packets

Why threads rather than processes: the work is I/O bound (Neo4j + HTTP), the
Neo4j driver is thread-safe, and a shared driver avoids re-establishing the
connection pool per worker. The usage tracker uses a ContextVar, which is
thread-local, so token accounting stays per question.

Resumability: one JSONL per (split, run tag), appended under a lock. A question
already present is skipped, so a killed sweep can be restarted with the same
--run-tag and it will pick up where it stopped.

Sharding: --shard-index / --shard-count partition the split by question index so
several machines (or several processes on one machine) can work the same split
without colliding on rows.

Usage:
    # pilot: 30 multi-hop questions, 16 in flight
    python tools/run_parallel.py --splits multi_hop --limit 30 --concurrency 16 \
        --run-tag pilot-30

    # full sweep, resumable
    python tools/run_parallel.py --splits single_hop,multi_hop,hallucination \
        --concurrency 32 --run-tag v41-flash-r1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
KGQA_DIR = PROJECT_DIR / "KG_QA"
sys.path.insert(0, str(KGQA_DIR))

from graph import HazardNeo4jClient  # noqa: E402
from search import (  # noqa: E402
    extract_query_slots,
    extract_topic_entities,
    repair_entity_packets,
    search_single_entity,
    unresolved_mention_packets,
)
from main import finalize_entity_packets, run_options  # noqa: E402
from utils import (  # noqa: E402
    configure_client,
    configure_http_pool,
    load_qa_csv,
    track_llm_usage,
)


def build_args(ns: argparse.Namespace) -> SimpleNamespace:
    """Mirror the attribute names KG_QA/search.py reads off `args`."""
    return SimpleNamespace(
        dataset="qa",
        input_csv=str(ns.input_csv),
        input_split=ns.splits,
        output_dir=str(ns.output_dir),
        start_question=1,
        samples=10 ** 9,
        single_question="",
        max_length=ns.max_length,
        temperature_exploration=0.0,
        temperature_reasoning=0.0,
        width=ns.width,
        depth=ns.depth,
        min_relation_score=0.3,
        min_candidate_score=0.3,
        top_k_paths_for_reasoning=5,
        candidate_pool_size=8,
        max_candidates_per_relation=12,
        LLM_type=ns.model,
        entity_model=ns.entity_model or ns.model,
        api_base=ns.api_base,
        opeani_api_keys=ns.api_key,
        ablation=ns.ablation,
        sufficiency_mode=ns.sufficiency_mode,
        structural_rule=ns.structural_rule,
        entity_guard=ns.entity_guard,
        repair=ns.repair,
        joint_pool_size=ns.joint_pool_size,
        judge_max_tokens=ns.judge_max_tokens,
        field_completion=ns.field_completion,
        slot_guided_expansion=ns.slot_guided_expansion,
        neo4j_uri=ns.neo4j_uri,
        neo4j_user=ns.neo4j_user,
        neo4j_password=ns.neo4j_password,
    )


def read_done(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a hard kill
            qid = rec.get("qid")
            if isinstance(qid, int):
                done.add(qid)
    return done


def run_one(
    qid: int,
    question: str,
    gold: str,
    args: SimpleNamespace,
    graph_client: HazardNeo4jClient,
) -> dict[str, Any]:
    started = time.time()
    with track_llm_usage() as usage:
        query_slots = extract_query_slots(question, args)
        entities = extract_topic_entities(question, graph_client, args, query_slots=query_slots)
        packets = [
            search_single_entity(question, entity, graph_client, args, query_slots=query_slots)
            for entity in entities
        ]
        packets, repair_log = repair_entity_packets(question, packets, graph_client, args, query_slots)
        # Parsed mentions the entity guard could not confirm in the graph are
        # chemicals without evidence; the aggregator then abstains.
        packets = packets + unresolved_mention_packets(query_slots)
        # Dispatch through the canonical aggregator rather than reimplementing
        # the branch here. An earlier version duplicated this logic inline, so
        # the wo_abstention switch in main.finalize_entity_packets was never
        # reached and that variant silently behaved like `full`.
        final, end_mode, remark = finalize_entity_packets(
            question, entities, packets, args
        )

    # A run with zero LLM requests means the pipeline silently degraded: every
    # slot came back empty, every packet came back insufficient, and the answer
    # is a template. Recording such a row as a normal result is what let an
    # earlier sweep look complete while never touching the API. Flag it instead.
    llm_calls = usage.request_count
    degraded = llm_calls == 0

    return {
        "qid": qid,
        "question": question,
        "gold": gold,
        "ablation": getattr(args, "ablation", "full"),
        "options": run_options(args),
        "slots": {k: v for k, v in query_slots.items() if not str(k).startswith("_")},
        "resolution": query_slots.get("_resolution", {}),
        "repair_log": repair_log,
        "entities": entities,
        "answer": final.get("answer", ""),
        "evidence": final.get("evidence", ""),
        "reasoning_paths": final.get("reasoning_paths", []),
        "end_mode": end_mode,
        # wo_abstention deliberately generates on partial evidence, so its
        # end_mode is a third state that must not be folded into either
        # "answered" or "abstained" -- otherwise the table would credit it with
        # coverage it does not have.
        "abstained": end_mode not in ("complete_then_aggregate",
                                      "all_entities_satisfied_then_aggregate"),
        "forced_generation": end_mode == "forced_generation_on_partial_evidence",
        "degraded_no_llm": degraded,
        "entity_packets": [
            {
                "entity": p.get("entity"),
                "status": p.get("status"),
                "clue": p.get("clue"),
                "triplets": p.get("triplets"),
                "answered_paths": [
                    {"triplets": ap.get("triplets"), "path_score": ap.get("path_score"),
                     "completed_sibling": bool(ap.get("completed_sibling", False))}
                    for ap in (p.get("answered_paths") or [])
                ],
                "dead_paths": [
                    {"triplets": dp.get("triplets")}
                    for dp in (p.get("dead_paths") or [])
                ],
                "path_score": p.get("path_score"),
                # Revision diagnostics: which paths the search generated and
                # what the judge said at each depth (E8/E11 error attribution).
                "sufficiency_mode": p.get("sufficiency_mode", "per_path"),
                "min_hops": p.get("min_hops"),
                "judge_log": p.get("judge_log", []),
                "explored_paths": (p.get("explored_paths") or [])[:80],
                "repaired": bool(p.get("repaired", False)),
            }
            for p in packets
        ],
        "usage": usage.to_dict(),
        "time_cost": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent SCEPS runner.")
    parser.add_argument("--input-csv", default=str(PROJECT_DIR / "Question&Answer"))
    parser.add_argument("--splits", default="multi_hop")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3,
                        help="retry a question this many times when it makes zero LLM calls")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--model", default=os.getenv("KGQA_MODEL", "cn:deepseek-v4.1-flash"))
    parser.add_argument("--ablation", default="full",
                        choices=["full", "wo_slot_coverage", "wo_sufficiency_check",
                                 "wo_abstention", "entity_target_only",
                                 "wo_constraint_slot"],
                        help="which SCEPS component configuration to run")
    parser.add_argument("--sufficiency-mode", default="joint", choices=["joint", "per_path"],
                        help="joint = one judgement per depth over the candidate pool (revised); "
                             "per_path = original one-path-at-a-time judge")
    parser.add_argument("--structural-rule", default="on", choices=["on", "off"],
                        help="a non-empty bridge slot requires a supporting path of >= 2 hops")
    parser.add_argument("--entity-guard", default="on", choices=["on", "off"],
                        help="conservative chemical-name resolution (KG_QA/entity_guard.py)")
    parser.add_argument("--repair", default="off", choices=["on", "off"],
                        help="one slot-repair-and-research round for insufficient entities")
    parser.add_argument("--joint-pool-size", type=int, default=20)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--field-completion", default="on", choices=["on", "off"],
                        help="add every value of the accepted target field (same node, same relation)")
    parser.add_argument("--slot-guided-expansion", default="on", choices=["on", "off"],
                        help="always expand the relation matching the target/bridge slot and keep "
                             "intermediates named in the question")
    parser.add_argument("--entity-model", default=os.getenv("KGQA_ENTITY_MODEL", ""))
    parser.add_argument("--api-base", default=os.getenv("KGQA_API_BASE", ""))
    parser.add_argument("--api-key", action="append", default=None,
                        help="repeatable; each key is used round-robin")
    parser.add_argument("--api-keys", default="",
                        help="comma-separated keys (alternative to repeating --api-key)")
    parser.add_argument("--requests-per-minute", type=int, default=0,
                        help="fixed request ceiling; 0 = unlimited")
    parser.add_argument("--adaptive", action="store_true",
                        help="enable AIMD throttling toward an unknown upstream quota")
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", ""))
    ns = parser.parse_args()

    # Resolve the key pool: repeated --api-key wins, then --api-keys, then the
    # environment variable.
    keys: list[str] = []
    if ns.api_key:
        keys.extend(k for k in ns.api_key if k.strip())
    if ns.api_keys:
        keys.extend(k.strip() for k in ns.api_keys.split(",") if k.strip())
    if not keys:
        env_key = os.getenv("KGQA_API_KEY", "")
        if env_key.strip():
            keys.append(env_key.strip())
    if not keys:
        sys.exit("need --api-key/--api-keys or KGQA_API_KEY")
    if ns.shard_count < 1 or not (0 <= ns.shard_index < ns.shard_count):
        sys.exit("invalid shard arguments")

    configure_client(keys, ns.requests_per_minute)
    # Unless a fixed ceiling is given, let the client find the real one. With
    # --concurrency 20 the workers can push ~150 req/min at 8s latency, which is
    # above the quota these runs actually see. AIMD backs off on 429 and creeps
    # back up on success, so the effective rate settles without anybody having
    # to know the upstream limit.
    # Adaptive throttling stays available but off by default: the upstream quota
    # is now high enough that 20 workers never approach it.
    if ns.adaptive:
        configure_adaptive_limit(ns.concurrency)
        print(f"adaptive throttling: ceiling {ns.concurrency} in-flight requests")
    configure_http_pool(ns.concurrency + 16)
    print(f"api keys: {len(keys)} (round-robin); "
          f"requests/min ceiling: {ns.requests_per_minute or 'unlimited'}; "
          f"http pool: {ns.concurrency + 16}")

    args = build_args(ns)
    args.opeani_api_keys = keys[0]  # the pool overrides per-call selection
    out_dir = Path(ns.output_dir) / ns.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_client = HazardNeo4jClient(
        ns.neo4j_uri, ns.neo4j_user, ns.neo4j_password,
        max_pool_size=ns.concurrency + 20,
    )

    write_lock = threading.Lock()
    started_at = time.time()
    grand_total = 0

    try:
        for split in [s.strip() for s in ns.splits.split(",") if s.strip()]:
            csv_path = Path(ns.input_csv) / f"{split}.csv"
            if not csv_path.is_file():
                print(f"skip {split}: {csv_path} not found")
                continue
            rows = load_qa_csv(csv_path)
            jsonl_path = out_dir / f"{split}.jsonl"
            done = read_done(jsonl_path)

            todo: list[tuple[int, str, str]] = []
            for idx, row in enumerate(rows, start=1):
                if idx < ns.start_index:
                    continue
                if idx % ns.shard_count != ns.shard_index:
                    continue
                if idx in done:
                    continue
                todo.append((idx, row["question"], row.get("reference_answer", "")))
            if ns.limit:
                todo = todo[: ns.limit]

            total = len(todo)
            print(f"\n=== {split}: {total} questions to run "
                  f"(skipped {len(done)} already done, concurrency {ns.concurrency}) ===")
            if not total:
                continue

            completed = 0
            failures = 0
            split_started = time.time()
            with jsonl_path.open("a", encoding="utf-8") as fh:
                with ThreadPoolExecutor(max_workers=ns.concurrency) as pool:
                    futures = {
                        pool.submit(run_one, qid, q, gold, args, graph_client): qid
                        for qid, q, gold in todo
                    }
                    for fut in as_completed(futures):
                        qid = futures[fut]
                        completed += 1
                        try:
                            record = fut.result()
                        except Exception as exc:  # keep the sweep alive
                            failures += 1
                            record = {
                                "qid": qid,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-1200:],
                            }
                        # A zero-LLM row is an infrastructure failure, not a result.
                        # Retry it immediately so a brief 503 burst does not cost
                        # a question; only give up after --retries attempts, and
                        # mark it so the analyzer can exclude it.
                        if record.get("degraded_no_llm"):
                            gold = next((g for q, _q2, g in todo if q == qid), "")
                            for attempt in range(1, ns.retries + 1):
                                backoff = min(2.0 ** attempt, 20.0)
                                time.sleep(backoff)
                                try:
                                    retry = run_one(qid, record.get("question", ""), gold,
                                                    args, graph_client)
                                except Exception:
                                    continue
                                if not retry.get("degraded_no_llm"):
                                    record = retry
                                    break
                            record.setdefault("retries_used", ns.retries)
                        with write_lock:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            fh.flush()
                        if completed % 10 == 0 or completed == total:
                            elapsed = time.time() - split_started
                            rate = completed / elapsed if elapsed else 0
                            eta = (total - completed) / rate if rate else 0
                            print(f"  {completed}/{total}  {rate:.2f} q/s  "
                                  f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  "
                                  f"failed {failures}")
            grand_total += completed
            print(f"  wrote {jsonl_path}")

            # Report degradation loudly: a split whose rows mostly carry zero LLM
            # calls did not actually exercise the system, whatever the file says.
            degraded = 0
            answered = 0
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("degraded_no_llm"):
                        degraded += 1
                    if not rec.get("abstained"):
                        answered += 1
            rows_written = completed + len(done)
            if degraded:
                print(f"  !! WARNING: {degraded}/{rows_written} rows made zero LLM calls "
                      f"in {split}. That means the API was unreachable, not that the "
                      f"system abstained. Do not report these numbers.")
            print(f"  {split}: answered {answered}/{rows_written} "
                  f"({100.0 * answered / max(rows_written, 1):.1f}%)")
    finally:
        graph_client.close()

    elapsed = time.time() - started_at
    print(f"\ndone: {grand_total} questions in {elapsed/60:.1f} min "
          f"({grand_total/elapsed:.2f} q/s)" if elapsed else "\ndone")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
