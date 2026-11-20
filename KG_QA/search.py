from __future__ import annotations

import math
from typing import Any

from entity_guard import build_name_index, guarded_resolve, merge_constraints
from graph import HazardNeo4jClient
from prompts import (
    CANDIDATE_PRUNE_SYSTEM,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_REASONING_SYSTEM,
    FINAL_AGGREGATION_SYSTEM,
    JOINT_SUFFICIENCY_SYSTEM,
    RELATION_PRUNE_SYSTEM,
    SLOT_REPAIR_SYSTEM,
    build_candidate_prune_prompt,
    build_entity_extraction_prompt,
    build_entity_reasoning_prompt,
    build_final_aggregation_prompt,
    build_joint_sufficiency_prompt,
    build_relation_prune_prompt,
    build_slot_repair_prompt,
)
from utils import call_chat_json, normalize_text, overlap_score, record_fallback, render_path_chain


# ------------------------------------------------------------- run options ---
#
# Revision switches (2026-11-17). Defaults are the revised method; each switch
# can be turned off to reproduce the earlier behaviour or to run an ablation.
#
#   sufficiency_mode  "joint" (default): one judgement per depth over the pool
#                     of candidate paths, the judge names the supporting
#                     subset. "per_path": the original one-path-at-a-time judge.
#   structural_rule   on (default): when the parsed bridge slot is non-empty,
#                     at least one supporting path must have >= 2 hops.
#   entity_guard      on (default): see entity_guard.py.
#   repair            off (default): one slot-repair-and-research round for
#                     entities that end without sufficient evidence.

def _option_flag(args, name: str, default: bool) -> bool:
    value = getattr(args, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "off", "no", "")
    return bool(value)


def sufficiency_mode(args) -> str:
    mode = str(getattr(args, "sufficiency_mode", "joint") or "joint").strip().lower()
    if mode not in ("joint", "per_path"):
        raise ValueError(f"unknown sufficiency_mode: {mode}")
    return mode


def structural_rule_on(args) -> bool:
    return _option_flag(args, "structural_rule", True)


def entity_guard_on(args) -> bool:
    return _option_flag(args, "entity_guard", True)


def repair_on(args) -> bool:
    return _option_flag(args, "repair", False)


def field_completion_on(args) -> bool:
    """After acceptance, add every value of the accepted target field (same node,
    same relation) as evidence. Revision 2026-11-18, see _complete_fields."""
    return _option_flag(args, "field_completion", True)


def slot_guided_on(args) -> bool:
    """Make the slots steer expansion deterministically (see _augment_relations
    and the anchoring step in _expand_frontier). Revision 2026-11-18."""
    return _option_flag(args, "slot_guided_expansion", True)


def required_hops(args, query_slots: dict[str, Any] | None) -> int:
    """Slot-derived structural requirement: a bridge slot implies >= 2 hops."""
    if not structural_rule_on(args):
        return 1
    bridge = str((query_slots or {}).get("bridge") or "").strip()
    return 2 if bridge else 1


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_candidate_name(name: str) -> str:
    """Equality key for matching an LLM-echoed candidate against the pool.

    Graph tail strings mix full- and half-width punctuation and differ in
    whitespace, so '氧化性液体,类别2' and '氧化性液体，类别 2' must compare equal.
    """
    cleaned = "".join(str(name or "").split())
    return (cleaned.replace("（", "(").replace("）", ")")
                   .replace("，", ",").replace("：", ":")
                   .replace("；", ";"))


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
        "completed_sibling": bool(record.get("completed_sibling", False)),
        "anchored": bool(record.get("anchored", False)),
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
        record_fallback("rank_no_api_key")
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
            # Relation ranking emits a short list of names; the reasoning trace
            # adds cost without changing the selected set.
            reasoning="none",
        )
    except Exception:
        record_fallback("relation_rank_error")
        return _fallback_rows()
    if not isinstance(payload, dict):
        record_fallback("relation_rank_unparsable")
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
        record_fallback("relation_rank_no_valid_item")
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
        record_fallback("rank_no_api_key")
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
            # Candidate filtering picks indices from a supplied list; same
            # argument as relation ranking.
            reasoning="none",
        )
    except Exception:
        record_fallback("candidate_rank_error")
        return _fallback_candidates()
    if not isinstance(payload, dict):
        record_fallback("candidate_rank_unparsable")
        return _fallback_candidates()

    selected = []
    seen = set()
    pool_by_name: dict[str, list[dict[str, Any]]] = {}
    for candidate in pruned_pool:
        pool_by_name.setdefault(candidate["name"], []).append(candidate)
    pool_by_norm: dict[str, list[dict[str, Any]]] = {}
    for candidate in pruned_pool:
        pool_by_norm.setdefault(_normalize_candidate_name(candidate["name"]), []).append(candidate)

    for item in _extract_ranked_llm_items(payload, "selected"):
        raw_name = str(item.get("name", "")).strip()
        # Accept the three shapes models actually emit for this prompt:
        #   "名称"
        #   "名称 | 化学名 -[关系]-> 名称"      (the rendered candidate line)
        #   "1. 名称"                          (the numbered list item)
        # Only the first was handled originally, so against a real graph every
        # call missed the pool and silently fell back to score ordering.
        matched: list[dict[str, Any]] = []
        for candidate_name in (raw_name, raw_name.split("|", 1)[0].strip(),
                               raw_name.split(".", 1)[-1].strip() if raw_name[:1].isdigit() else ""):
            if not candidate_name:
                continue
            if candidate_name in pool_by_name:
                matched = pool_by_name[candidate_name]
                break
            normalized = _normalize_candidate_name(candidate_name)
            if normalized in pool_by_norm:
                matched = pool_by_norm[normalized]
                break
        if not matched:
            # Last resort: the model may have echoed the full triplet, so match
            # on any pool entry whose triplet appears verbatim in the reply.
            matched = [c for c in pruned_pool if c["triplet"] and c["triplet"] in raw_name]
        for candidate in matched:
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
        record_fallback("candidate_rank_no_valid_item")
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


# ------------------------------------------------------------------ ablations ---
#
# Six configurations selectable at run time. Every switch is applied at a named
# decision point so the difference between two variants is auditable:
#
#   full                  all five slots must be covered; LLM judges sufficiency;
#                         abstains when no complete evidence exists.
#   wo_slot_coverage      acceptance becomes target-hit only: the sufficiency
#                         judge is replaced by a string test on the target field.
#                         No LLM call, no slot-coverage requirement.
#   wo_sufficiency_check  acceptance becomes a structural heuristic (path length
#                         >= 2). Differs from the previous switch in what it
#                         replaces: coverage is still parsed but never tested.
#   wo_abstention         when no complete evidence is found, the generator is
#                         still called on whatever partial evidence exists
#                         instead of returning an insufficient-evidence answer.
#   entity_target_only    only chemical and target slots are required; event,
#                         bridge, and constraint are dropped at parse time.
#   wo_constraint_slot    constraint is dropped at parse time; the other four
#                         slots are still enforced.
#
# Placement matters: the two parse-time switches change what the LLM is asked
# to cover, while the two acceptance switches leave parsing intact and change
# only the accept/reject rule. That distinction is what the ablation table in
# the paper reports.

VALID_ABLATIONS = {
    "full",
    "wo_slot_coverage",
    "wo_sufficiency_check",
    "wo_abstention",
    "entity_target_only",
    "wo_constraint_slot",
}


def ablation_of(args) -> str:
    value = getattr(args, "ablation", "full") or "full"
    if value not in VALID_ABLATIONS:
        raise ValueError(f"unknown ablation: {value}")
    return value


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

    ablation = ablation_of(args)

    # wo_sufficiency_check: replace the LLM judgement with a structural
    # heuristic. A path of two or more hops is accepted; a single hop is not.
    # No API call is made, so this variant is also the cheap one.
    if ablation == "wo_sufficiency_check":
        return {
            "sufficient": path_hops >= 2,
            "missing": "" if path_hops >= 2 else clue,
        }

    if not args.opeani_api_keys:
        # Offline demo mode only (no API key configured); counted so that a
        # real run can never report numbers produced this way unnoticed.
        record_fallback("judge_no_api_key")
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
            # Same budget as the joint judge (revision 2026-11-19): with 768 tokens the
            # per-path ablation lost 58 judgements to truncation, which confounded
            # the joint-vs-per-path comparison.
            max_tokens=max(int(args.max_length), int(getattr(args, "judge_max_tokens", 4096) or 4096)),
            # Deliberately left at the backend default: this is the sufficiency
            # judgement the whole framework rests on, so the model keeps its full
            # reasoning budget. The asymmetry with the parsing and ranking calls
            # is intentional and reported in the paper.
            reasoning="auto",
        )
    except Exception:
        # A failed judgement is "insufficient", never "accepted". The earlier
        # code accepted any path of >= 2 hops here, which contradicted the
        # paper's statement that invalid judgements count as insufficient.
        record_fallback("judge_error")
        return {"sufficient": False, "missing": clue}
    if not isinstance(payload, dict):
        record_fallback("judge_unparsable")
        return {"sufficient": False, "missing": clue}

    sufficient = bool(payload.get("sufficient"))

    # wo_slot_coverage: keep the LLM call (so the variant still has the same
    # cost profile and the same parsed slots) but override its verdict with a
    # target-field string test. This isolates "requires full slot coverage"
    # from "uses an LLM judge": the two are switched off in different variants.
    if ablation == "wo_slot_coverage":
        target = str((query_slots or {}).get("target") or "").strip()
        rendered = " ".join(paths)
        sufficient = bool(target) and (target in rendered)

    return {
        "sufficient": sufficient,
        "missing": str(payload.get("missing", "")).strip(),
    }


# ----------------------------------------------------- joint sufficiency ---

def _parse_support(value: Any, pool_size: int) -> list[int]:
    """1-based indices from the judge -> unique, valid 0-based indices."""
    items = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
    support: list[int] = []
    for item in items:
        try:
            idx = int(str(item).strip()) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < pool_size and idx not in support:
            support.append(idx)
    return support


def _judge_joint_sufficiency(
    *,
    question: str,
    entity_name: str,
    pool: list[dict[str, Any]],
    clue: str,
    args,
    query_slots: dict[str, Any] | None,
    min_hops: int,
) -> dict[str, Any]:
    """One judgement over a pool of candidate paths for one chemical.

    Returns a dict with ``sufficient``, ``support`` (0-based pool indices),
    ``missing``, ``slot_support``, ``reject_reason`` and ``llm_called``.
    Acceptance requires (i) the judge's ``sufficient`` verdict, (ii) at least
    one valid supporting index, and (iii) a supporting path that meets the
    slot-derived hop requirement. Any failure is "insufficient".
    """
    result: dict[str, Any] = {
        "sufficient": False, "support": [], "missing": clue,
        "slot_support": {}, "reject_reason": "", "llm_called": False,
    }
    if not pool:
        result["reject_reason"] = "empty_pool"
        return result
    hops = [len(path.get("triplets", [])) for path in pool]
    eligible = [i for i, h in enumerate(hops) if h >= min_hops]
    if not eligible:
        bridge = str((query_slots or {}).get("bridge") or "").strip()
        result.update(missing=clue or (f"需要经过中间环节：{bridge}" if bridge else ""),
                      reject_reason="structural_precheck")
        return result

    top_k = max(1, int(getattr(args, "top_k_paths_for_reasoning", 5) or 5))
    ablation = ablation_of(args)
    by_score = sorted(eligible, key=lambda i: _coerce_score(pool[i].get("path_score")), reverse=True)

    # wo_sufficiency_check: no semantic judgement at all; accept the best
    # paths that satisfy the structural (hop) requirement.
    if ablation == "wo_sufficiency_check":
        result.update(sufficient=True, support=by_score[:top_k], missing="")
        return result

    # wo_slot_coverage: acceptance is a literal test on the target slot.
    if ablation == "wo_slot_coverage":
        target = normalize_text(str((query_slots or {}).get("target") or ""))
        support = [
            i for i in by_score
            if target and target in normalize_text(render_path_chain(pool[i].get("triplets", [])))
        ]
        if support:
            result.update(sufficient=True, support=support[:top_k], missing="")
        else:
            result["reject_reason"] = "target_string_not_found"
        return result

    if not args.opeani_api_keys:
        record_fallback("judge_no_api_key")
        result.update(sufficient=True, support=by_score[:top_k], reject_reason="offline_no_api_key")
        return result

    rendered = [render_path_chain(path.get("triplets", [])) for path in pool]
    result["llm_called"] = True
    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=JOINT_SUFFICIENCY_SYSTEM,
            user_prompt=build_joint_sufficiency_prompt(
                question=question,
                entity_name=entity_name,
                paths=rendered,
                clue=clue,
                query_slots=query_slots,
                min_hops=min_hops,
            ),
            temperature=args.temperature_reasoning,
            # The judge keeps the backend's default reasoning budget, so the
            # cap must leave room for hidden reasoning tokens; a truncated
            # reply is counted (finish_reason) and treated as insufficient.
            max_tokens=max(int(args.max_length), int(getattr(args, "judge_max_tokens", 4096) or 4096)),
            reasoning="auto",
        )
    except Exception:
        record_fallback("judge_error")
        result["reject_reason"] = "judge_error"
        return result
    if not isinstance(payload, dict):
        record_fallback("judge_unparsable")
        result["reject_reason"] = "judge_unparsable"
        return result

    verdict = payload.get("sufficient")
    sufficient = verdict is True or str(verdict).strip().lower() == "true"
    missing = str(payload.get("missing", "") or "").strip()
    slot_support = payload.get("slot_support") if isinstance(payload.get("slot_support"), dict) else {}
    support = _parse_support(payload.get("support"), len(pool))
    result.update(missing=missing or clue, slot_support=slot_support)
    if not sufficient:
        result["reject_reason"] = "judged_insufficient"
        return result
    if not support:
        record_fallback("judge_sufficient_without_support")
        result["reject_reason"] = "no_valid_support"
        return result
    if not any(hops[i] >= min_hops for i in support):
        result.update(reject_reason="structural_rule", missing=missing or "支持路径未经过问题要求的中间环节")
        return result
    result.update(sufficient=True, support=support)
    return result


# --------------------------------------------- slot-guided expansion (11-18) ---
#
# Error analysis of the first revised run (multi-hop): 14 questions reached the
# right intermediate node but the relation holding the asked field was never
# expanded, and 41 never reached the intermediate that the question itself
# names (e.g. "被划为易燃液体,类别3"). The slots are therefore used to steer the
# expansion deterministically, in addition to being shown to the LLM ranker:
#   * the relation that best matches the target slot (and, at depth 1, the
#     bridge slot) is always expanded;
#   * at depth 1, a candidate whose name occurs verbatim in the question is
#     always kept and placed first in the next frontier.

def _strip_rel_prefix(name: str) -> str:
    return name.split(":", 1)[1] if name.startswith("属性:") else name


def _slot_relation_score(slot: str, relation: str, event: str = "") -> float:
    slot_n = _normalize_candidate_name(slot)
    rel_n = _normalize_candidate_name(_strip_rel_prefix(relation))
    if not slot_n or not rel_n:
        return 0.0
    score = overlap_score(slot_n, rel_n)
    if slot_n in rel_n or rel_n in slot_n:
        score += 1.0
    event_n = _normalize_candidate_name(event)
    if event_n and event_n in rel_n:
        score += 0.5  # e.g. event 吸入 picks 吸入:急救措施 over 皮肤接触:急救措施
    return score


def _augment_relations(
    selected: list[dict[str, Any]],
    available: list[dict[str, Any]],
    query_slots: dict[str, Any] | None,
    depth: int,
) -> list[dict[str, Any]]:
    slots = query_slots or {}
    wanted = []
    if depth == 1 and str(slots.get("bridge") or "").strip():
        wanted.append(str(slots["bridge"]))
    if str(slots.get("target") or "").strip():
        wanted.append(str(slots["target"]))
    event = str(slots.get("event") or "")
    names = {row["relation"] for row in selected}
    out = list(selected)
    for slot in wanted:
        scored = sorted(available, key=lambda row: _slot_relation_score(slot, row["relation"], event), reverse=True)
        if not scored:
            continue
        best = scored[0]
        # Only a containment match counts; partial character overlap is too weak.
        if _slot_relation_score(slot, best["relation"], event) >= 1.0 and best["relation"] not in names:
            out.append({**best, "relation_score": 0.6, "local_relation_score": 0.0, "slot_guided": True})
            names.add(best["relation"])
    return out


def _complete_fields(
    answered: list[dict[str, Any]],
    graph_client: HazardNeo4jClient,
    args,
) -> list[dict[str, Any]]:
    """Field completion: for every accepted path, add the other values of its
    last relation at the same source node.

    The judge accepts the field (node + relation) that answers the question,
    but candidate selection keeps only the top `width` values of a relation,
    so multi-valued fields (precaution lists, first-aid steps, extinguishing
    agents) reached the generator truncated: 21 multi-hop questions in the
    first revised run were on the right route but missed some gold parts.
    Siblings share node and relation with an accepted path, so no new
    semantic decision is made; they complete the slot the judge accepted.
    """
    out = list(answered)
    seen = {tuple(p.get("triplets", [])) for p in answered}
    limit = int(getattr(args, "max_candidates_per_relation", 12) or 12)
    for path in answered:
        parent = path.get("parent_node") or {}
        relation = path.get("relation_raw") or ""
        if not parent.get("element_id") or not relation:
            continue
        if hasattr(graph_client, "expand_descriptor"):
            rows = graph_client.expand_descriptor(parent["element_id"], relation, limit)
        else:
            rows = graph_client.expand_relation(parent["element_id"], relation, limit)
        rel_name = _strip_rel_prefix(relation)
        for row in rows:
            value = str(row.get("name", "") or "").strip() or "<空>"
            triplets = list(path.get("triplets", []))[:-1] + [f"{parent.get('name', '')} -[{rel_name}]-> {value}"]
            key = tuple(triplets)
            if key in seen:
                continue
            seen.add(key)
            out.append({**path, "triplets": triplets, "path_status": "answered", "completed_sibling": True,
                        "path_score": round(_coerce_score(path.get("path_score")) * 0.999, 6)})
    return out


def _expand_frontier(
    *,
    question: str,
    entity_name: str,
    frontier: list[dict[str, Any]],
    depth: int,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand every frontier path by one graph fact (same mechanics as the
    per-path search). Returns (new path records, name-only dead paths)."""
    new_records: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    guided = slot_guided_on(args)
    question_norm = _normalize_candidate_name(question)
    for current_path in frontier:
        current_node = current_path["current_node"]
        path_clue = str(current_path.get("clue", "") or "")
        available_relations = _get_available_descriptor_rows(graph_client, current_node["element_id"])
        if _is_name_only_followup(current_path, available_relations):
            dead.append({**current_path, "path_status": "dead", "clue": path_clue})
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
        if guided:
            selected_relations = _augment_relations(selected_relations, available_relations, query_slots, depth)
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
            if guided and depth == 1:
                # Anchor intermediates that the question names verbatim.
                kept = {c["triplet"] for c in selected_candidates}
                for candidate in candidate_payloads:
                    name_norm = _normalize_candidate_name(candidate["name"])
                    if len(name_norm) >= 3 and name_norm in question_norm and candidate["triplet"] not in kept:
                        selected_candidates = list(selected_candidates) + [
                            {**candidate, "candidate_score": 0.95, "local_candidate_score": 0.0, "anchored": True}
                        ]
                        kept.add(candidate["triplet"])
            for candidate in selected_candidates:
                candidate_score = _coerce_score(candidate.get("candidate_score"))
                next_step_scores = current_path["step_scores"] + [
                    {
                        "triplet": candidate["triplet"],
                        "relation_score": relation_score,
                        "candidate_score": candidate_score,
                        "step_score": _compute_step_score(relation_score, candidate_score),
                    }
                ]
                target_node = candidate["node"]
                target_id = target_node.get("element_id")
                new_records.append(
                    {
                        "triplets": current_path["triplets"] + [candidate["triplet"]],
                        "current_node": target_node,
                        "parent_node": current_node,
                        "relation_raw": relation_name,
                        "depth": depth,
                        "path_score": _compute_path_score(next_step_scores),
                        "step_scores": next_step_scores,
                        "visited_ids": set(current_path["visited_ids"]) | ({target_id} if target_id else set()),
                        "path_status": "active",
                        "clue": path_clue,
                        "can_expand": _path_can_expand(graph_client, current_path, target_node),
                        "anchored": bool(candidate.get("anchored")) or bool(current_path.get("anchored")),
                        "slot_guided": bool(relation_row.get("slot_guided")),
                    }
                )
    return new_records, dead


def _joint_pool(
    generated: dict[tuple[str, ...], dict[str, Any]],
    new_records: list[dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    """Paths shown to the judge: the newest depth first, then earlier paths."""
    newest = _sort_path_records(new_records)
    seen = {tuple(r.get("triplets", [])) for r in newest}
    older = [r for r in _sort_path_records(list(generated.values())) if tuple(r.get("triplets", [])) not in seen]
    pool = newest[:cap]
    if len(pool) < cap:
        pool += older[: cap - len(pool)]
    return pool


def _search_single_entity_joint(
    question: str,
    entity_name: str,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None = None,
    query_plan: Any | None = None,
) -> dict[str, Any]:
    root_node = graph_client.get_chemical_node(entity_name)
    if root_node is None:
        return {
            "entity": entity_name, "triplets": [], "evidence_lines": [], "search_trace": [],
            "status": "missing_entity", "path_score": 0.0, "scored_paths": [], "answered_paths": [],
            "dead_paths": [], "active_paths_at_end": [], "clue": "", "verbose_paths": [],
            "sufficiency_mode": "joint", "judge_log": [], "explored_paths": [],
        }

    min_hops = required_hops(args, query_slots)
    pool_cap = max(1, int(getattr(args, "joint_pool_size", 20) or 20))
    frontier: list[dict[str, Any]] = [
        {
            "current_node": root_node, "triplets": [], "step_scores": [], "depth": 0,
            "path_score": 1.0, "visited_ids": {root_node["element_id"]},
            "path_status": "active", "clue": "",
        }
    ]
    generated: dict[tuple[str, ...], dict[str, Any]] = {}
    dead_path_map: dict[tuple[str, ...], dict[str, Any]] = {}
    search_trace: list[dict[str, Any]] = []
    judge_log: list[dict[str, Any]] = []
    answered_paths: list[dict[str, Any]] = []
    clue = ""

    for depth in range(1, args.depth + 1):
        new_records, name_only_dead = _expand_frontier(
            question=question, entity_name=entity_name, frontier=frontier, depth=depth,
            graph_client=graph_client, args=args, query_slots=query_slots,
        )
        for record in name_only_dead:
            _upsert_path_map(dead_path_map, record)
        for record in new_records:
            _upsert_path_map(generated, record)
        search_trace.append(
            {"depth": depth, "expansions": [_serialize_path_record(r) for r in new_records], "clue": clue}
        )
        if not new_records:
            break

        pool = _joint_pool(generated, new_records, pool_cap)
        verdict = _judge_joint_sufficiency(
            question=question, entity_name=entity_name, pool=pool, clue=clue,
            args=args, query_slots=query_slots, min_hops=min_hops,
        )
        judge_log.append(
            {
                "depth": depth,
                "pool_size": len(pool),
                "sufficient": bool(verdict["sufficient"]),
                "support": [render_path_chain(pool[i].get("triplets", [])) for i in verdict["support"]],
                "slot_support": verdict.get("slot_support", {}),
                "missing": str(verdict.get("missing", "") or "")[:300],
                "reject_reason": verdict.get("reject_reason", ""),
                "llm_called": bool(verdict.get("llm_called")),
            }
        )
        clue = str(verdict.get("missing", "") or clue)
        if verdict["sufficient"]:
            answered_paths = [{**pool[i], "path_status": "answered"} for i in verdict["support"]]
            if field_completion_on(args):
                before = len(answered_paths)
                answered_paths = _complete_fields(answered_paths, graph_client, args)
                judge_log[-1]["completed_siblings"] = len(answered_paths) - before
            break

        expandable: list[dict[str, Any]] = []
        for record in new_records:
            if record.get("can_expand"):
                expandable.append({**record, "clue": clue})
            else:
                _upsert_path_map(dead_path_map, {**record, "path_status": "dead"})
        ranked = _sort_path_records(expandable)
        if slot_guided_on(args):
            # Intermediates named in the question go first (see _expand_frontier).
            ranked = [r for r in ranked if r.get("anchored")] + [r for r in ranked if not r.get("anchored")]
        frontier = ranked[: args.width]
        if not frontier:
            break

    packet = _build_entity_packet(
        entity_name=entity_name,
        search_trace=search_trace,
        status="answered" if answered_paths else "insufficient",
        clue=clue,
        answered_paths=_sort_path_records(answered_paths),
        dead_paths=_sort_path_records(list(dead_path_map.values())),
        active_paths_at_end=[] if answered_paths else _sort_path_records(frontier),
        query_plan=query_plan,
    )
    packet.update(
        {
            "sufficiency_mode": "joint",
            "min_hops": min_hops,
            "judge_log": judge_log,
            "explored_paths": [
                render_path_chain(r.get("triplets", []))
                for r in _sort_path_records(list(generated.values()))
            ][:80],
        }
    )
    return packet


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
        record_fallback("parse_no_api_key")
        return slots
    payload = None
    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.entity_model or args.LLM_type,
            system_prompt=ENTITY_EXTRACTION_SYSTEM,
            user_prompt=build_entity_extraction_prompt(question),
            temperature=0.0,
            # Reasoning-capable backends spend max_tokens on hidden reasoning
            # tokens before emitting any content. With a 256 cap a slot-parse
            # call returns finish_reason=length and an empty string, which then
            # silently degrades into "no slots, no entity, abstain" for every
            # question. Budget for the reasoning trace, not just the answer.
            max_tokens=max(args.max_length, 1024),
            # Slot parsing is pure extraction into a fixed schema.
            reasoning="none",
        )
        if not isinstance(payload, dict):
            record_fallback("parse_unparsable")
    except Exception:
        record_fallback("parse_error")
        payload = None
    if not isinstance(payload, dict):
        return slots
    slots["chemicals"] = _coerce_entity_list(payload.get("chemicals"))
    slots["event"] = str(payload.get("event", "") or "").strip()
    slots["bridge"] = str(payload.get("bridge", "") or "").strip()
    slots["target"] = str(payload.get("target", "") or "").strip()
    slots["constraint"] = _coerce_constraint_list(payload.get("constraint"))

    # Parse normalisation (revision 2026-11-18). The prompt requires a target,
    # but the parser sometimes puts the asked field into `bridge` and leaves
    # `target` empty (e.g. "X的一个燃烧产物是什么" -> bridge=燃烧产物). With the
    # structural rule on, that inconsistent parse would demand a two-hop path
    # for a one-hop question, so the lone bridge is treated as the target.
    if slots["bridge"] and (
        not slots["target"] or normalize_text(slots["target"]) == normalize_text(slots["bridge"])
    ):
        if not slots["target"]:
            slots["target"] = slots["bridge"]
        slots["bridge"] = ""
        slots["normalization"] = "bridge_moved_to_target"

    # Parse-time ablation switches. Clearing a slot here removes it from the
    # requirement set the sufficiency judge is asked about, which is exactly
    # what "we only require chemical and target" means operationally.
    ablation = ablation_of(args)
    if ablation == "entity_target_only":
        slots["event"] = ""
        slots["bridge"] = ""
        slots["constraint"] = []
    elif ablation == "wo_constraint_slot":
        slots["constraint"] = []

    return slots


# Ablations in which the constraint slot must stay empty; the entity guard
# must not re-introduce a constraint behind their back.
_NO_CONSTRAINT_ABLATIONS = {"wo_constraint_slot", "entity_target_only"}


def resolve_topic_entities(
    question: str,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Guarded resolution (see entity_guard.py). Parsed mentions come first;
    question-level exact matching is only a fallback when the parser returned
    no chemical, and never uses single-character names."""
    if query_slots is None:
        query_slots = extract_query_slots(question, args)
    info: dict[str, Any] = {"entities": [], "constraints": [], "unresolved": [], "log": [], "source": "none"}
    mentions = _coerce_entity_list(query_slots.get("chemicals"))
    if mentions:
        index = build_name_index(graph_client.list_name_records())
        res = guarded_resolve(mentions, index)
        info.update(entities=res.entities, constraints=res.constraints, unresolved=res.unresolved,
                    log=res.log, source="parsed_mentions")
        if res.entities or res.unresolved:
            # Do not fall back to substring matching over the question when a
            # mention was parsed but rejected: that fallback is precisely what
            # turned 溴化乙锭 into 溴.
            return info
    matcher = getattr(graph_client, "find_exact_chemical_mentions", None)
    matches = [name for name in (matcher(question) if callable(matcher) else []) if len(name) >= 2]
    if matches:
        info.update(entities=matches, source="question_match_fallback")
    return info


def extract_topic_entities(
    question: str,
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None = None,
) -> list[str]:
    if query_slots is None:
        query_slots = extract_query_slots(question, args)

    if entity_guard_on(args):
        info = resolve_topic_entities(question, graph_client, args, query_slots=query_slots)
        if info["constraints"] and ablation_of(args) not in _NO_CONSTRAINT_ABLATIONS:
            query_slots["constraint"] = merge_constraints(
                _coerce_entity_list(query_slots.get("constraint")), info["constraints"]
            )
        # Kept on the slot dict (ignored by every prompt builder) so that the
        # runners can write it to the record and abstain for unresolved mentions.
        query_slots["_resolution"] = {k: info[k] for k in ("entities", "constraints", "unresolved", "log", "source")}
        return info["entities"]

    # Original resolver (entity_guard off), kept for reproducing earlier runs.
    exact_matcher = getattr(graph_client, "find_exact_chemical_mentions", None)
    exact_matches = exact_matcher(question) if callable(exact_matcher) else []
    if exact_matches:
        return exact_matches

    raw_entities = _coerce_entity_list(query_slots.get("chemicals"))
    resolved = graph_client.resolve_chemical_names(raw_entities)
    if resolved:
        return resolved
    return graph_client.local_entity_fallback(question)


def unresolved_mention_packets(query_slots: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Placeholder packets for parsed chemical mentions the guard could not
    confirm in the graph. Adding them makes the final step abstain, in line
    with the rule that every chemical in the question needs evidence."""
    info = (query_slots or {}).get("_resolution") or {}
    return [
        {
            "entity": mention, "triplets": [], "evidence_lines": [], "search_trace": [],
            "status": "unresolved_mention", "path_score": 0.0, "scored_paths": [],
            "answered_paths": [], "dead_paths": [], "active_paths_at_end": [],
            "clue": "该化学品未能在图谱中确认", "verbose_paths": [],
        }
        for mention in info.get("unresolved", [])
    ]


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
    if sufficiency_mode(args) == "joint":
        return _search_single_entity_joint(
            question, entity_name, graph_client, args, query_slots=query_slots, query_plan=query_plan
        )
    # ---- original per-path search (sufficiency_mode="per_path") ----
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
        limit = top_k
        if packet.get("sufficiency_mode") == "joint":
            # The judge named these paths as jointly sufficient (plus completed
            # siblings of the same field); dropping some of them before
            # generation would undo that decision. Cap at 16.
            limit = max(top_k, min(len(packet.get("answered_paths", [])), 16))
        packet_copy["answered_paths"] = _limit_paths_for_reasoning(packet.get("answered_paths", []), limit)
        reasoning_packets.append(packet_copy)
    reasoning_paths = []
    for packet in reasoning_packets:
        for path in packet.get("answered_paths", []):
            chain = render_path_chain(path.get("triplets", []))
            if chain:
                reasoning_paths.append(f"{packet['entity']}: {chain}")
    if not args.opeani_api_keys:
        record_fallback("generation_no_api_key")
        return _fallback_aggregate_answer(reasoning_packets)

    payload = None
    try:
        payload = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=FINAL_AGGREGATION_SYSTEM,
            user_prompt=build_final_aggregation_prompt(question, reasoning_packets),
            temperature=args.temperature_reasoning,
            # Same reasoning-budget argument as the judge: the default
            # max_length (768) can be consumed by hidden reasoning.
            max_tokens=max(int(args.max_length), int(getattr(args, "judge_max_tokens", 4096) or 4096)),
            # Final composition over already-accepted evidence: keep the default
            # reasoning budget so the answer stays faithful to the paths.
            reasoning="auto",
        )
        if not isinstance(payload, dict):
            record_fallback("generation_unparsable")
    except Exception:
        record_fallback("generation_error")
        payload = None
    if not isinstance(payload, dict):
        # Template answer built only from accepted evidence (no LLM). Counted
        # above so that a run can report how often this happened.
        return _fallback_aggregate_answer(reasoning_packets)
    return {
        "answer": str(payload.get("answer", "")).strip(),
        "evidence": str(payload.get("evidence", "")).strip(),
        "reasoning_paths": reasoning_paths,
    }


# ----------------------------------------------------------------- repair ---

def repair_entity_packets(
    question: str,
    entity_packets: list[dict[str, Any]],
    graph_client: HazardNeo4jClient,
    args,
    query_slots: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One bounded correction round (off unless ``args.repair`` is set).

    For every entity that ended without sufficient evidence, the slot parse is
    re-examined against the fields that actually exist on the chemical node,
    using the judge's ``missing`` note as feedback. If the corrected slots
    differ, that entity is searched once more. Constraints from the question
    are never dropped. Returns (packets, repair_log).
    """
    if not repair_on(args) or not args.opeani_api_keys:
        return entity_packets, []
    repaired: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for packet in entity_packets:
        if packet.get("status") != "insufficient":
            repaired.append(packet)
            continue
        entity = packet.get("entity", "")
        root = graph_client.get_chemical_node(entity)
        if root is None:
            repaired.append(packet)
            continue
        fields = []
        for row in _get_available_descriptor_rows(graph_client, root["element_id"]):
            name = str(row.get("relation", ""))
            name = name.split(":", 1)[1] if name.startswith("属性:") else name
            if name and name not in fields:
                fields.append(name)
        entry: dict[str, Any] = {"entity": entity, "before": {k: (query_slots or {}).get(k) for k in ("event", "bridge", "target", "constraint")}}
        try:
            payload = call_chat_json(
                api_base=args.api_base,
                api_key=args.opeani_api_keys,
                model=args.entity_model or args.LLM_type,
                system_prompt=SLOT_REPAIR_SYSTEM,
                user_prompt=build_slot_repair_prompt(
                    question=question,
                    entity_name=entity,
                    query_slots=query_slots,
                    missing=str(packet.get("clue", "") or ""),
                    available_fields=fields,
                ),
                temperature=0.0,
                max_tokens=max(int(args.max_length), 1024),
                reasoning="none",
            )
        except Exception:
            record_fallback("repair_error")
            payload = None
        if not isinstance(payload, dict):
            entry["outcome"] = "repair_call_failed"
            log.append(entry)
            repaired.append(packet)
            continue
        new_slots = dict(query_slots or {})
        for key in ("event", "bridge", "target"):
            new_slots[key] = str(payload.get(key, new_slots.get(key, "")) or "").strip()
        # Constraints come from the question and the entity guard; keep them.
        new_slots["constraint"] = list((query_slots or {}).get("constraint") or [])
        entry["after"] = {k: new_slots.get(k) for k in ("event", "bridge", "target", "constraint")}
        if all(entry["after"][k] == entry["before"].get(k) for k in ("event", "bridge", "target")):
            entry["outcome"] = "unchanged"
            log.append(entry)
            repaired.append(packet)
            continue
        retry = search_single_entity(question, entity, graph_client, args, query_slots=new_slots)
        entry["outcome"] = "answered_after_repair" if retry.get("status") == "answered" else "still_insufficient"
        log.append(entry)
        if retry.get("status") == "answered":
            retry["repaired"] = True
            retry["repair_slots"] = entry["after"]
            repaired.append(retry)
        else:
            repaired.append(packet)
    return repaired, log
