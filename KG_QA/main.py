from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graph import HazardNeo4jClient
from search import (
    aggregate_entity_packets,
    build_incomplete_aggregation,
    extract_query_slots,
    extract_topic_entities,
    repair_entity_packets,
    search_single_entity,
    unresolved_mention_packets,
)
from utils import dump_json, load_qa_csv, save_results_csv, slice_rows, track_llm_usage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM_QA over Neo4j hazardous chemical KG.")
    default_output_dir = str(Path(__file__).resolve().parent.parent / "output")
    parser.add_argument("--dataset", type=str, default="qa")
    parser.add_argument("--input_csv", type=str, default="../Question&Answer")
    parser.add_argument("--input_split", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--start_question", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--single_question", type=str, default="")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--temperature_exploration", type=float, default=0.0)
    parser.add_argument("--temperature_reasoning", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--min_relation_score", type=float, default=0.3)
    parser.add_argument("--min_candidate_score", type=float, default=0.3)
    parser.add_argument("--top_k_paths_for_reasoning", type=int, default=5)
    parser.add_argument("--candidate_pool_size", type=int, default=8)
    parser.add_argument("--max_candidates_per_relation", type=int, default=12)
    parser.add_argument("--LLM_type", type=str, default=os.getenv("KGQA_MODEL", "gpt-4o"))
    parser.add_argument("--entity_model", type=str, default=os.getenv("KGQA_ENTITY_MODEL", ""))
    parser.add_argument("--api_base", type=str, default=os.getenv("KGQA_API_BASE", os.getenv("OPENAI_BASE_URL", "")))
    parser.add_argument("--opeani_api_keys", type=str, default=os.getenv("KGQA_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--neo4j_uri", type=str, default=os.getenv("KGQA_NEO4J_URI", os.getenv("NEO4J_URI", "")))
    parser.add_argument("--neo4j_user", type=str, default=os.getenv("KGQA_NEO4J_USER", os.getenv("NEO4J_USER", "")))
    parser.add_argument("--neo4j_password", type=str, default=os.getenv("KGQA_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "")))
    # Revision switches (2026-11-17); see the "run options" block in search.py.
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "wo_slot_coverage", "wo_sufficiency_check", "wo_abstention",
                                 "entity_target_only", "wo_constraint_slot"])
    parser.add_argument("--sufficiency_mode", type=str, default="joint", choices=["joint", "per_path"])
    parser.add_argument("--structural_rule", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--entity_guard", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--repair", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--joint_pool_size", type=int, default=20)
    parser.add_argument("--judge_max_tokens", type=int, default=4096)
    parser.add_argument("--field_completion", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--slot_guided_expansion", type=str, default="on", choices=["on", "off"])
    return parser


def run_options(args) -> dict[str, Any]:
    """The configuration that determines a run's behaviour, written to every record."""
    return {
        "model": getattr(args, "LLM_type", ""),
        "entity_model": getattr(args, "entity_model", "") or getattr(args, "LLM_type", ""),
        "ablation": getattr(args, "ablation", "full"),
        "sufficiency_mode": getattr(args, "sufficiency_mode", "joint"),
        "structural_rule": getattr(args, "structural_rule", "on"),
        "entity_guard": getattr(args, "entity_guard", "on"),
        "repair": getattr(args, "repair", "off"),
        "field_completion": getattr(args, "field_completion", "on"),
        "slot_guided_expansion": getattr(args, "slot_guided_expansion", "on"),
        "width": getattr(args, "width", None),
        "depth": getattr(args, "depth", None),
        "joint_pool_size": getattr(args, "joint_pool_size", 20),
        "judge_max_tokens": getattr(args, "judge_max_tokens", 4096),
        "max_length": getattr(args, "max_length", None),
    }


def _resolve_rows(args) -> list[dict[str, str]]:
    if args.single_question.strip():
        return [{"question": args.single_question.strip(), "reference_answer": ""}]
    rows = load_qa_csv(args.input_csv)
    return slice_rows(rows, args.start_question, args.samples)


def _normalize_input_split(value: str | list[str] | tuple[str, ...] | None) -> str | list[str]:
    if isinstance(value, (list, tuple)):
        normalized_values = [
            str(item).strip().lower().replace("-", "_")
            for item in value
            if str(item).strip()
        ]
        if not normalized_values:
            return "all"
        if any(item == "all" for item in normalized_values):
            return "all"
        deduped: list[str] = []
        seen: set[str] = set()
        for item in normalized_values:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    normalized = str(value or "all").strip().lower().replace("-", "_")
    return normalized or "all"


def _resolve_datasets(args) -> list[tuple[str, list[dict[str, str]]]]:
    if args.single_question.strip():
        return [("single_question", [{"question": args.single_question.strip(), "reference_answer": ""}])]

    input_path = Path(args.input_csv)
    input_split = _normalize_input_split(getattr(args, "input_split", "all"))
    if input_path.is_dir():
        datasets: list[tuple[str, list[dict[str, str]]]] = []
        csv_paths = sorted(input_path.glob("*.csv"))
        if input_split != "all":
            requested_splits = input_split if isinstance(input_split, list) else [input_split]
            csv_path_map = {csv_path.stem: csv_path for csv_path in csv_paths}
            missing = [split for split in requested_splits if split not in csv_path_map]
            if missing:
                missing_text = ", ".join(f"{split}.csv" for split in missing)
                raise FileNotFoundError(f"未在 {input_path} 中找到对应输入数据集: {missing_text}")
            csv_paths = [csv_path_map[split] for split in requested_splits]
        for csv_path in csv_paths:
            rows = slice_rows(load_qa_csv(csv_path), args.start_question, args.samples)
            datasets.append((csv_path.stem, rows))
        return datasets

    rows = slice_rows(load_qa_csv(input_path), args.start_question, args.samples)
    return [(input_path.stem, rows)]


def finalize_entity_packets(
    question: str,
    topic_entities: list[str],
    entity_packets: list[dict[str, Any]],
    args,
    query_plan: Any | None = None,
) -> tuple[dict[str, str], str, str]:
    if not topic_entities:
        return (
            {"answer": "未抽取到有效化学品实体。", "evidence": "", "reasoning_paths": []},
            "no_entity_extracted",
            "entity extraction failed",
        )

    answered_packets = [packet for packet in entity_packets if packet["status"] == "answered"]
    if len(answered_packets) == len(entity_packets):
        final_result = aggregate_entity_packets(question, answered_packets, args, query_plan=query_plan)
        if query_plan is not None:
            final_result = {
                **final_result,
                "complete": bool(final_result.get("complete", True)),
                "abstain": bool(final_result.get("abstain", False)),
                "missing_slots": list(final_result.get("missing_slots", [])),
                "stop_reason": str(final_result.get("stop_reason", "event_chain_complete")),
            }
            return (
                final_result,
                "complete_then_aggregate",
                str(final_result.get("stop_reason", "event_chain_complete")),
            )
        return (
            final_result,
            "all_entities_satisfied_then_aggregate",
            "ok",
        )

    # wo_abstention: instead of returning the insufficient-evidence template,
    # call the generator on whatever partial evidence was collected. This is the
    # variant that shows what the abstention rule is buying -- the answers below
    # are exactly the fluent-but-unsupported outputs the framework is designed
    # to suppress, so the variant is expected to raise hallucination and lower
    # nothing else.
    ablation = str(getattr(args, "ablation", "full") or "full")
    if ablation == "wo_abstention":
        partial = aggregate_entity_packets(question, entity_packets, args, query_plan=query_plan)
        return (
            partial,
            "forced_generation_on_partial_evidence",
            "abstention disabled for ablation",
        )

    return (
        build_incomplete_aggregation(entity_packets, args),
        "partial_or_insufficient",
        "some entities did not reach sufficient evidence",
    )


def run_pipeline(args, rows: list[dict[str, str]] | None = None, dataset_name: str | None = None) -> list[dict[str, Any]]:
    if rows is None:
        rows = _resolve_rows(args)
    graph_client = HazardNeo4jClient(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    all_results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=args.start_question):
            question = row["question"]
            reference_answer = row.get("reference_answer", "")
            started_at = time.time()

            with track_llm_usage() as llm_usage:
                query_slots = extract_query_slots(question, args)
                topic_entities = extract_topic_entities(question, graph_client, args, query_slots=query_slots)
                entity_packets = [
                    search_single_entity(question, entity, graph_client, args, query_slots=query_slots)
                    for entity in topic_entities
                ]
                entity_packets, repair_log = repair_entity_packets(
                    question, entity_packets, graph_client, args, query_slots
                )
                # Parsed mentions the entity guard could not confirm count as
                # chemicals without evidence, so the question abstains.
                entity_packets = entity_packets + unresolved_mention_packets(query_slots)
                final_result, end_mode, remark = finalize_entity_packets(
                    question, topic_entities, entity_packets, args
                )
            best_path_score = max((float(packet.get("path_score", 0.0) or 0.0) for packet in entity_packets), default=0.0)

            result_row = {
                "index": index,
                "question": question,
                "reference_answer": reference_answer,
                "options": run_options(args),
                "slots": {k: v for k, v in query_slots.items() if not str(k).startswith("_")},
                "resolution": query_slots.get("_resolution", {}),
                "repair_log": repair_log,
                "topic_entities": topic_entities,
                "entity_packets": entity_packets,
                "answer": final_result["answer"],
                "evidence": final_result["evidence"],
                "reasoning_paths": final_result.get("reasoning_paths", []),
                "end_mode": end_mode,
                "remark": remark,
                "time_cost": round(time.time() - started_at, 3),
                "best_path_score": round(best_path_score, 6),
                "llm_usage": llm_usage.to_dict(),
                "llm_traces": llm_usage.traces,
            }
            all_results.append(result_row)
            print(f"[{index}] entities={topic_entities} end_mode={end_mode}")
    finally:
        graph_client.close()
    return all_results


def save_outputs(results: list[dict[str, Any]], args, dataset_name: str | None = None) -> None:
    if dataset_name is None and hasattr(args, "output_json") and hasattr(args, "output_csv"):
        output_json = Path(args.output_json)
        output_csv = Path(args.output_csv)
    else:
        dataset_name = dataset_name or "qa_results"
        output_dir = Path(getattr(args, "output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"{dataset_name}.json"
        output_csv = output_dir / f"{dataset_name}.csv"

    json_payload = [
        {
            "question": item["question"],
            "ground_truth": item["reference_answer"],
            "options": item.get("options", {}),
            "slots": item.get("slots", {}),
            "resolution": item.get("resolution", {}),
            "repair_log": item.get("repair_log", []),
            "topic_entities": item["topic_entities"],
            "answer": item["answer"],
            "evidence": item["evidence"],
            "reasoning_paths": item.get("reasoning_paths", []),
            "end_mode": item["end_mode"],
            "remark": item["remark"],
            "time_cost": item["time_cost"],
            "best_path_score": item.get("best_path_score", 0.0),
            "llm_usage": item["llm_usage"],
            "llm_traces": item.get("llm_traces", []),
            "entity_packets": item["entity_packets"],
        }
        for item in results
    ]
    dump_json(output_json, json_payload)

    csv_rows = [
        {
            "题目": item["question"],
            "参考答案": item["reference_answer"],
            "LLM答案": item["answer"],
            "LLM依据": item["evidence"],
            "依据路径": "\n".join(item.get("reasoning_paths", [])),
            "消耗时间": item["time_cost"],
            "LLM请求次数": item.get("llm_usage", {}).get("request_count", 0),
            "消耗总token": item.get("llm_usage", {}).get("total_tokens", 0),
        }
        for item in results
    ]
    save_results_csv(output_csv, csv_rows)


def main(argv: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    parser = build_parser()
    args = parser.parse_args(argv)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, rows in _resolve_datasets(args):
        results = run_pipeline(args, rows=rows, dataset_name=dataset_name)
        save_outputs(results, args, dataset_name)
        all_results[dataset_name] = results
    return all_results


def run_from_config(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(config)
    args = SimpleNamespace(**defaults)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, rows in _resolve_datasets(args):
        results = run_pipeline(args, rows=rows, dataset_name=dataset_name)
        save_outputs(results, args, dataset_name)
        all_results[dataset_name] = results
    return all_results


if __name__ == "__main__":
    main()
