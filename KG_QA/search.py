from __future__ import annotations

import math
from typing import Any

from graph import HazardNeo4jClient
from prompts import (
    CANDIDATE_PRUNE_SYSTEM,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_REASONING_SYSTEM,
    FINAL_AGGREGATION_SYSTEM,
    RELATION_PRUNE_SYSTEM,
    build_candidate_prune_prompt,
    build_entity_extraction_prompt,
    build_entity_reasoning_prompt,
    build_final_aggregation_prompt,
    build_relation_prune_prompt,
)
from utils import call_chat_json, normalize_text, overlap_score, render_path_chain


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_ranked_llm_items(payload: dict[str, Any], list_key: str) -> list[dict[str, Any]]:
    raw_items = payload.get(list_key)
    if isinstance(raw_items, list):
        items = raw_items
    else:
        items = []
        for key, value in payload.items():
            if key == list_key:
                continue
            if isinstance(value, dict):
                items.append(value)

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (
            str(
                item.get("name")
                or item.get("candidate")
                or item.get("候选项")
                or item.get("候选")
                or ""
            ).strip()
        )
        score = item.get("score", item.get("分数"))
        if not name:
            continue
        normalized.append({"name": name, "score": _coerce_score(score)})
    return normalized


def _sort_path_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            _coerce_score(item.get("path_score")),
            len(item.get("triplets", [])),
            " | ".join(item.get("triplets", [])),
        ),
        reverse=True,
    )


def _select_display_best_path(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            len(item.get("triplets", [])),
            _coerce_score(item.get("path_score")),
            " | ".join(item.get("triplets", [])),
        ),
    )


def _compute_step_score(relation_score: float, candidate_score: float) -> float:
    return round(math.sqrt(max(relation_score, 0.0) * max(candidate_score, 0.0)), 6)


def _compute_path_score(step_scores: list[dict[str, Any]], epsilon: float = 1e-8) -> float:
    if not step_scores:
        return 1.0
    log_sum = 0.0
    for item in step_scores:
        step_score = _coerce_score(item.get("step_score"))
        log_sum += math.log(step_score + epsilon)
    return round(math.exp(log_sum / len(step_scores)), 6)


def _serialize_path_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "triplets": list(record.get("triplets", [])),
        "current_node": record.get("current_node"),
        "depth": int(record.get("depth", 0) or 0),
        "path_score": _coerce_score(record.get("path_score")),
        "step_scores": list(record.get("step_scores", [])),
        "path_status": str(record.get("path_status", "active")),
        "clue": str(record.get("clue", "") or ""),
    }


def _normalize_compare_text(value: str) -> str:
    return normalize_text(value or "")


def _derive_query_slots(query_plan: Any | None) -> dict[str, Any] | None:
    if query_plan is None:
        return None
    return {
        "chemicals": list(getattr(query_plan, "chemicals", []) or []),
        "event": str(getattr(query_plan, "event", "") or "").strip(),
        "bridge": str(getattr(query_plan, "bridge", "") or "").strip(),
        "target": str(getattr(query_plan, "target", "") or "").strip(),
        "constraint": list(getattr(query_plan, "constraint", []) or []),
    }


def _build_verbose_paths(paths: list[dict[str, Any]], query_plan: Any | None) -> list[dict[str, Any]]:
    verbose_paths = []
    for path in paths:
        triplets = list(path.get("triplets", []))
        verbose_paths.append(
            {
                "triplets": triplets,
                "path_score": _coerce_score(path.get("path_score")),
                "score_breakdown": {
                    "step_scores": list(path.get("step_scores", [])),
                    "path_score": _coerce_score(path.get("path_score")),
                },
            }
        )
    return verbose_paths


def _limit_paths_for_reasoning(paths: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return _sort_path_records(list(paths))[:top_k]


def _upsert_path_map(path_map: dict[tuple[str, ...], dict[str, Any]], record: dict[str, Any]) -> None:
    path_key = tuple(record.get("triplets", []))
    existing = path_map.get(path_key)
    if existing is None or _coerce_score(record.get("path_score")) >= _coerce_score(existing.get("path_score")):
        path_map[path_key] = record


def _get_available_descriptor_rows(graph_client: HazardNeo4jClient, element_id: str) -> list[dict[str, Any]]:
    if hasattr(graph_client, "get_available_descriptors"):
        return graph_client.get_available_descriptors(element_id)
    return graph_client.get_available_relations(element_id)


def _path_can_expand(
    graph_client: HazardNeo4jClient,
    current_path: dict[str, Any],
    target_node: dict[str, Any],
) -> bool:
    target_id = target_node.get("element_id")
    if not target_id:
        return False
    if target_id in current_path.get("visited_ids", set()):
        return False
    return bool(_get_available_descriptor_rows(graph_client, target_id))


def _is_name_only_followup(
    current_path: dict[str, Any],
    available_relations: list[dict[str, Any]],
) -> bool:
    if int(current_path.get("depth", 0) or 0) <= 0:
        return False
    if not available_relations:
        return False
    relation_names = [str(row.get("relation", "")).strip() for row in available_relations]
    return all(name == "属性:name" for name in relation_names)


def _build_entity_packet(
    *,
    entity_name: str,
    search_trace: list[dict[str, Any]],
    status: str,
    clue: str,
    answered_paths: list[dict[str, Any]],
    dead_paths: list[dict[str, Any]],
    active_paths_at_end: list[dict[str, Any]],
    query_plan: Any | None = None,
) -> dict[str, Any]:
    preferred_pool = answered_paths if answered_paths else active_paths_at_end or dead_paths
    best_path = _select_display_best_path(preferred_pool)
    display_pool = preferred_pool
    if best_path:
        best_depth = len(best_path.get("triplets", []))
        display_pool = [item for item in preferred_pool if len(item.get("triplets", [])) == best_depth][:8]

    verbose_pool = answered_paths if answered_paths else active_paths_at_end or dead_paths
    return {
        "entity": entity_name,
        "triplets": best_path["triplets"] if best_path else [],
        "evidence_lines": best_path["triplets"] if best_path else [],
        "search_trace": search_trace,
        "status": status,
        "path_score": _coerce_score(best_path["path_score"]) if best_path else 0.0,
        "scored_paths": [_serialize_path_record(item) for item in display_pool],
        "answered_paths": [_serialize_path_record(item) for item in answered_paths],
        "dead_paths": [_serialize_path_record(item) for item in dead_paths],
        "active_paths_at_end": [_serialize_path_record(item) for item in active_paths_at_end],
        "clue": clue,
        "verbose_paths": _build_verbose_paths(verbose_pool, query_plan),
    }


def _local_relation_prefilter(question: str, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in relations:
        rel_name = str(row["relation"])
        match_name = rel_name.split(":", 1)[1] if rel_name.startswith("属性:") else rel_name
        score = overlap_score(question, match_name)
        if match_name in question:
            score += 1.0
        scored.append({**row, "_local_score": score})
    scored.sort(key=lambda item: item["_local_score"], reverse=True)
    return scored


def _select_relations(
    *,
    question: str,
    root_entity: str,
    current_node: dict[str, Any],
    path_summary: str,
    clue: str,
    relation_rows: list[dict[str, Any]],
    args,
    query_slots: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pruned_pool = _local_relation_prefilter(question, relation_rows)
    if not pruned_pool:
        return []

    def _fallback_rows() -> list[dict[str, Any]]:
        selected_rows = []
        for row in pruned_pool[: args.width]:
            relation_score = _coerce_score(row.get("_local_score"))
            selected_rows.append(
                {
                    **row,
                    "relation_score": relation_score,
                    "local_relation_score": _coerce_score(row.get("_local_score")),
                }
            )
        return selected_rows

    if not args.opeani_api_keys:
        return _fallback_rows()

    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=RELATION_PRUNE_SYSTEM,
            user_prompt=build_relation_prune_prompt(
                question=question,
                root_entity=root_entity,
                current_node=current_node,
                path_summary=path_summary,
                clue=clue,
                relation_rows=pruned_pool,
                width=args.width,
                query_slots=query_slots,
            ),
            temperature=args.temperature_exploration,
            max_tokens=args.max_length,
        )
    except Exception:
        return _fallback_rows()
    if not isinstance(payload, dict):
        return _fallback_rows()

    pool_by_name = {row["relation"]: row for row in pruned_pool}
    selected: list[dict[str, Any]] = []
    seen = set()
    for item in _extract_ranked_llm_items(payload, "relations"):
        name = str(item.get("name", "")).strip()
        if name and name in pool_by_name and name not in seen:
            row = pool_by_name[name]
            selected.append(
                {
                    **row,
                    "relation_score": _coerce_score(item.get("score")),
                    "local_relation_score": _coerce_score(row.get("_local_score")),
                }
            )
            seen.add(name)
        if len(selected) >= args.width:
            break
    if not selected:
        return _fallback_rows()
    selected.sort(
        key=lambda row: (
            _coerce_score(row.get("relation_score")),
            _coerce_score(row.get("local_relation_score")),
        ),
        reverse=True,
    )
    min_relation_score = _coerce_score(getattr(args, "min_relation_score", 0.3), 0.3)
    return [row for row in selected[: args.width] if _coerce_score(row.get("relation_score")) >= min_relation_score]


def _build_candidate_payloads(
    *,
    source_node: dict[str, Any],
    relation_name: str,
    expanded_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads = []
    source_name = source_node.get("name", "")
    for row in expanded_rows:
        target_name = str(row.get("name", "")).strip() or "<空>"
        reference = str(row.get("reference") or "").strip()
        triplet = f"{source_name} -[{relation_name}]-> {target_name}"
        payloads.append(
            {
                "node": {
                    "element_id": row["element_id"],
                    "label": str(row.get("label") or ""),
                    "name": target_name,
                },
                "name": target_name,
                "label": str(row.get("label") or ""),
                "triplet": triplet,
                "evidence_text": triplet,
                "reference": reference,
            }
        )
    return payloads


def _local_candidate_prefilter(question: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    scored = []
    for candidate in candidates:
        score = overlap_score(question, candidate["triplet"])
        if candidate["name"] and candidate["name"] in question:
            score += 1.0
        scored.append({**candidate, "_local_score": score})
    scored.sort(key=lambda item: item["_local_score"], reverse=True)
    return scored[:top_n]


def _select_candidates(
    *,
    question: str,
    root_entity: str,
    relation_name: str,
    path_summary: str,
    candidates: list[dict[str, Any]],
    args,
    query_slots: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pruned_pool = _local_candidate_prefilter(question, candidates, args.candidate_pool_size)
    if not pruned_pool:
        return []

    def _fallback_candidates() -> list[dict[str, Any]]:
        selected_rows = []
        for candidate in pruned_pool[: args.width]:
            candidate_score = _coerce_score(candidate.get("_local_score"))
            selected_rows.append(
                {
                    **candidate,
                    "candidate_score": candidate_score,
                    "local_candidate_score": _coerce_score(candidate.get("_local_score")),
                }
            )
        return selected_rows

    if not args.opeani_api_keys:
        return _fallback_candidates()

    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=CANDIDATE_PRUNE_SYSTEM,
            user_prompt=build_candidate_prune_prompt(
                question=question,
                root_entity=root_entity,
                relation_name=relation_name,
                path_summary=path_summary,
                candidates=pruned_pool,
                top_k=args.width,
                query_slots=query_slots,
            ),
            temperature=args.temperature_exploration,
            max_tokens=args.max_length,
        )
    except Exception:
        return _fallback_candidates()
    if not isinstance(payload, dict):
        return _fallback_candidates()

    selected = []
    seen = set()
    pool_by_name: dict[str, list[dict[str, Any]]] = {}
    for candidate in pruned_pool:
        pool_by_name.setdefault(candidate["name"], []).append(candidate)

    for item in _extract_ranked_llm_items(payload, "selected"):
        name = str(item.get("name", "")).strip()
        for candidate in pool_by_name.get(name, []):
            if candidate["triplet"] not in seen:
                selected.append(
                    {
                        **candidate,
                        "candidate_score": _coerce_score(item.get("score")),
                        "local_candidate_score": _coerce_score(candidate.get("_local_score")),
                    }
                )
                seen.add(candidate["triplet"])
                break
        if len(selected) >= args.width:
            break
    if not selected:
        return _fallback_candidates()
    selected.sort(
        key=lambda row: (
            _coerce_score(row.get("candidate_score")),
            _coerce_score(row.get("local_candidate_score")),
        ),
        reverse=True,
    )
    min_candidate_score = _coerce_score(getattr(args, "min_candidate_score", 0.3), 0.3)
    return [row for row in selected[: args.width] if _coerce_score(row.get("candidate_score")) >= min_candidate_score]


def _judge_entity_sufficiency(
    *,
    question: str,
    entity_name: str,
    paths: list[str],
    clue: str,
    path_hops: int,
    args,
    query_slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not paths:
        return {"sufficient": False, "missing": ""}
    if not args.opeani_api_keys:
        return {
            "sufficient": path_hops >= max(1, min(args.width, 2)),
            "missing": "",
        }

    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=ENTITY_REASONING_SYSTEM,
            user_prompt=build_entity_reasoning_prompt(
                question=question,
                entity_name=entity_name,
                paths=paths,
                clue=clue,
                query_slots=query_slots,
            ),
            temperature=args.temperature_reasoning,
            max_tokens=args.max_length,
        )
    except Exception:
        return {
            "sufficient": path_hops >= max(1, min(args.width, 2)),
            "missing": clue,
        }
    if not isinstance(payload, dict):
        return {"sufficient": False, "missing": clue}
    return {
        "sufficient": bool(payload.get("sufficient")),
        "missing": str(payload.get("missing", "")).strip(),
    }


def _coerce_entity_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def extract_query_slots(question: str, args) -> dict[str, Any]:
    slots = {
        "chemicals": [],
        "event": "",
        "bridge": "",
        "target": "",
        "constraint": [],
    }
    def _coerce_entity_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return []

    def _coerce_constraint_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return []

    if not args.opeani_api_keys:
        return slots
    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.entity_model or args.LLM_type,
            system_prompt=ENTITY_EXTRACTION_SYSTEM,
            user_prompt=build_entity_extraction_prompt(question),
            temperature=0.0,
            max_tokens=min(args.max_length, 256),
        )
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return slots
    slots["chemicals"] = _coerce_entity_list(payload.get("chemicals"))
    slots["event"] = str(payload.get("event", "") or "").strip()
    slots["bridge"] = str(payload.get("bridge", "") or "").strip()
    slots["target"] = str(payload.get("target", "") or "").strip()
    slots["constraint"] = _coerce_constraint_list(payload.get("constraint"))
    return slots


def extract_topic_entities(
    question: str,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None = None,
) -> list[str]:
    exact_matcher = getattr(graph_client, "find_exact_chemical_mentions", None)
    exact_matches = exact_matcher(question) if callable(exact_matcher) else []
    if exact_matches:
        return exact_matches

    if query_slots is None:
        query_slots = extract_query_slots(question, args)

    raw_entities = _coerce_entity_list(query_slots.get("chemicals"))
    resolved = graph_client.resolve_chemical_names(raw_entities)
    if resolved:
        return resolved
    return graph_client.local_entity_fallback(question)


def search_single_entity(
    question: str,
    entity_name: str,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None = None,
    query_plan: Any | None = None,
) -> dict[str, Any]:
    if query_slots is None and query_plan is not None:
        query_slots = _derive_query_slots(query_plan)
    root_node = graph_client.get_chemical_node(entity_name)
    if root_node is None:
        return {
            "entity": entity_name,
            "triplets": [],
            "evidence_lines": [],
            "search_trace": [],
            "status": "missing_entity",
            "path_score": 0.0,
            "scored_paths": [],
            "answered_paths": [],
            "dead_paths": [],
            "active_paths_at_end": [],
            "clue": "",
            "verbose_paths": [],
        }

    frontier = [
        {
            "current_node": root_node,
            "triplets": [],
            "step_scores": [],
            "depth": 0,
            "path_score": 1.0,
            "visited_ids": {root_node["element_id"]},
            "path_status": "active",
            "clue": "",
        }
    ]
    search_trace = []
    answered_path_map: dict[tuple[str, ...], dict[str, Any]] = {}
    dead_path_map: dict[tuple[str, ...], dict[str, Any]] = {}
    clue = ""

    for depth in range(1, args.depth + 1):
        depth_expansions = []
        next_frontier_map: dict[tuple[str, ...], dict[str, Any]] = {}
        for current_path in frontier:
            current_node = current_path["current_node"]
            path_clue = str(current_path.get("clue", "") or "")
            available_relations = _get_available_descriptor_rows(graph_client, current_node["element_id"])
            if _is_name_only_followup(current_path, available_relations):
                pruned_path = {
                    **current_path,
                    "path_status": "dead",
                    "clue": path_clue,
                }
                _upsert_path_map(dead_path_map, pruned_path)
                depth_expansions.append(pruned_path)
                clue = path_clue
                continue
            selected_relations = _select_relations(
                question=question,
                root_entity=entity_name,
                current_node=current_node,
                path_summary=render_path_chain(current_path["triplets"]),
                clue=path_clue,
                relation_rows=available_relations,
                args=args,
                query_slots=query_slots,
            )
            for relation_row in selected_relations:
                relation_name = relation_row["relation"]
                relation_score = _coerce_score(relation_row.get("relation_score"))
                if hasattr(graph_client, "expand_descriptor"):
                    expanded_rows = graph_client.expand_descriptor(
                        current_node["element_id"], relation_name, args.max_candidates_per_relation
                    )
                else:
                    expanded_rows = graph_client.expand_relation(
                        current_node["element_id"], relation_name, args.max_candidates_per_relation
                    )
                normalized_relation_name = relation_name.split(":", 1)[1] if relation_name.startswith("属性:") else relation_name
                candidate_payloads = _build_candidate_payloads(
                    source_node=current_node,
                    relation_name=normalized_relation_name,
                    expanded_rows=expanded_rows,
                )
                selected_candidates = _select_candidates(
                    question=question,
                    root_entity=entity_name,
                    relation_name=normalized_relation_name,
                    path_summary=render_path_chain(current_path["triplets"]),
                    candidates=candidate_payloads,
                    args=args,
                    query_slots=query_slots,
                )
                for candidate in selected_candidates:
                    candidate_score = _coerce_score(candidate.get("candidate_score"))
                    step_score = _compute_step_score(relation_score, candidate_score)
                    next_step_scores = current_path["step_scores"] + [
                        {
                            "triplet": candidate["triplet"],
                            "relation_score": relation_score,
                            "candidate_score": candidate_score,
                            "step_score": step_score,
                        }
                    ]
                    path_score = _compute_path_score(next_step_scores)
                    target_node = candidate["node"]
                    path_record = {
                        "triplets": current_path["triplets"] + [candidate["triplet"]],
                        "current_node": target_node,
                        "depth": depth,
                        "path_score": path_score,
                        "step_scores": next_step_scores,
                        "visited_ids": set(current_path["visited_ids"]),
                        "path_status": "active",
                        "clue": path_clue,
                    }
                    reasoning_input = [f"路径1: {render_path_chain(path_record['triplets'])}"]
                    reasoning = _judge_entity_sufficiency(
                        question=question,
                        entity_name=entity_name,
                        paths=reasoning_input,
                        clue=path_clue,
                        path_hops=len(path_record["triplets"]),
                        args=args,
                        query_slots=query_slots,
                    )
                    updated_path = {
                        **path_record,
                        "clue": str(reasoning.get("missing", "") or path_clue),
                    }
                    can_expand = _path_can_expand(graph_client, current_path, target_node)

                    if reasoning["sufficient"]:
                        updated_path["path_status"] = "answered"
                        _upsert_path_map(answered_path_map, updated_path)
                        clue = updated_path["clue"]
                    elif can_expand:
                        updated_path["path_status"] = "active"
                        target_id = target_node.get("element_id")
                        if target_id:
                            updated_path["visited_ids"] = set(current_path["visited_ids"]) | {target_id}
                        _upsert_path_map(next_frontier_map, updated_path)
                        clue = updated_path["clue"]
                    else:
                        updated_path["path_status"] = "dead"
                        _upsert_path_map(dead_path_map, updated_path)
                        clue = updated_path["clue"]

                    depth_expansions.append(updated_path)

        search_trace.append(
            {
                "depth": depth,
                "expansions": [_serialize_path_record(item) for item in depth_expansions],
                "clue": clue,
            }
        )

        next_frontier = _sort_path_records(list(next_frontier_map.values()))
        frontier = next_frontier[: args.width]
        if not frontier:
            break

    answered_paths = _sort_path_records(list(answered_path_map.values()))
    dead_paths = _sort_path_records(list(dead_path_map.values()))
    active_paths_at_end = _sort_path_records(frontier)
    if answered_paths:
        return _build_entity_packet(
            entity_name=entity_name,
            search_trace=search_trace,
            status="answered",
            clue=clue,
            answered_paths=answered_paths,
            dead_paths=dead_paths,
            active_paths_at_end=active_paths_at_end,
            query_plan=query_plan,
        )

    return _build_entity_packet(
        entity_name=entity_name,
        search_trace=search_trace,
        status="insufficient",
        clue=clue,
        answered_paths=[],
        dead_paths=dead_paths,
        active_paths_at_end=active_paths_at_end,
        query_plan=query_plan,
    )


def build_incomplete_aggregation(entity_packets: list[dict[str, Any]], args=None) -> dict[str, str]:
    answered_packets = [packet for packet in entity_packets if packet.get("status") == "answered"]
    incomplete_packets = [packet for packet in entity_packets if packet.get("status") != "answered"]
    top_k = max(1, int(getattr(args, "top_k_paths_for_reasoning", 5) or 5)) if args is not None else 5

    incomplete_names = "、".join(packet["entity"] for packet in incomplete_packets)
    answered_names = "、".join(packet["entity"] for packet in answered_packets)

    if answered_packets:
        answer = (
            "部分实体已完成检索，但仍有实体未达到足够证据，当前不执行最终汇总。"
            f"已完成实体：{answered_names}。未完成实体：{incomplete_names}。"
        )
        evidence_blocks = []
        for packet in answered_packets:
            block = [f"实体: {packet['entity']}"]
            answered_paths = _limit_paths_for_reasoning(packet.get("answered_paths", []), top_k)
            if answered_paths:
                for path in answered_paths:
                    block.append(f"证据: {render_path_chain(path.get('triplets', []))}")
            else:
                for line in packet.get("evidence_lines", [])[:top_k]:
                    block.append(f"证据: {line}")
            evidence_blocks.append("\n".join(block))
        return {"answer": answer, "evidence": "\n\n".join(evidence_blocks), "reasoning_paths": []}


    answer = (
        "根据当前检索结果，仍有实体未达到足够证据，当前不执行最终汇总。"
        f"未完成实体：{incomplete_names}。"
    )
    return {"answer": answer, "evidence": "", "reasoning_paths": []}


def _fallback_aggregate_answer(entity_packets: list[dict[str, Any]]) -> dict[str, str]:
    reasoning_paths = []
    for packet in entity_packets:
        for path in packet.get("answered_paths", []):
            chain = render_path_chain(path.get("triplets", []))
            if chain:
                reasoning_paths.append(f"{packet['entity']}: {chain}")

    if len(entity_packets) == 1:
        packet = entity_packets[0]
        answered_paths = packet.get("answered_paths", [])
        if answered_paths:
            evidence_lines = answered_paths[0].get("triplets", [])[:5]
        else:
            evidence_lines = packet.get("evidence_lines", [])[:5]
        if evidence_lines:
            return {
                "answer": "根据当前检索到的证据，相关信息如下：" + "；".join(evidence_lines),
                "evidence": "\n".join(evidence_lines),
                "reasoning_paths": reasoning_paths,
            }
        return {"answer": "根据当前检索结果，证据不足，未找到足够证据回答该问题。", "evidence": "", "reasoning_paths": []}

    lines = []
    for packet in entity_packets:
        answered_paths = packet.get("answered_paths", [])
        if answered_paths:
            evidence_lines = answered_paths[0].get("triplets", [])[:3]
        else:
            evidence_lines = packet.get("evidence_lines", [])[:3]
        if evidence_lines:
            lines.append(f"{packet['entity']}: {'；'.join(evidence_lines)}")
        else:
            lines.append(f"{packet['entity']}: 无足够证据")
    return {"answer": "；".join(lines), "evidence": "\n".join(lines), "reasoning_paths": reasoning_paths}


def aggregate_entity_packets(question: str, entity_packets: list[dict[str, Any]], args, query_plan: Any | None = None) -> dict[str, str]:
    if not entity_packets:
        return {"answer": "未抽取到有效化学品实体。", "evidence": "", "reasoning_paths": []}
    reasoning_packets = []
    top_k = max(1, int(getattr(args, "top_k_paths_for_reasoning", 5) or 5))
    for packet in entity_packets:
        packet_copy = dict(packet)
        packet_copy["answered_paths"] = _limit_paths_for_reasoning(packet.get("answered_paths", []), top_k)
        reasoning_packets.append(packet_copy)
    reasoning_paths = []
    for packet in reasoning_packets:
        for path in packet.get("answered_paths", []):
            chain = render_path_chain(path.get("triplets", []))
            if chain:
                reasoning_paths.append(f"{packet['entity']}: {chain}")
    if not args.opeani_api_keys:
        return _fallback_aggregate_answer(reasoning_packets)

    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=FINAL_AGGREGATION_SYSTEM,
            user_prompt=build_final_aggregation_prompt(question, reasoning_packets),
            temperature=args.temperature_reasoning,
            max_tokens=args.max_length,
        )
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return _fallback_aggregate_answer(reasoning_packets)
    return {
        "answer": str(payload.get("answer", "")).strip(),
        "evidence": str(payload.get("evidence", "")).strip(),
        "reasoning_paths": reasoning_paths,
    }
