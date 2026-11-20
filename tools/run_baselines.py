#!/usr/bin/env python
"""
Baseline KGQA methods for the hazardous-waste graph.

Four baselines, all over the same graph, same backbone, same token budget, same
half of the search budget as SCEPS. Each writes records in the same schema as
tools/run_parallel.py so one analyzer covers everything.

  kaping            retrieve graph facts around the question entities, rank
                    them, inject the top-k into a single generation prompt.
                    No iteration, no path structure. (Baek et al., 2023)

  tog               iterative beam search: at each step the LLM ranks the
                    relations of the current node, then picks which returned
                    candidates to keep. At the depth limit the best paths are
                    handed to the generator. (Sun et al., 2024)

  tog2              same search as ToG, but generation is knowledge-guided: the
                    accepted paths are rendered as an explicit reasoning chain
                    and the model is told to answer only through it.
                    (Ma et al., 2025)

  search_on_graph   observe-then-navigate: the LLM sees the full relation list
                    of the current node and chooses one; it then sees the
                    returned candidates and chooses one. It may stop and answer
                    at any step. No beam, single trajectory. (Sun et al., 2025)

  zero_shot / few_shot / cot
                    LLM-only prompting, no graph access. few_shot uses four
                    fixed examples built from chemicals that appear in none of
                    the 700 test questions or answers (FEW_SHOT_EXAMPLES).

Deliberate design decisions, all of which the paper must state:

  * Every graph baseline answers through _answer_prompt, which asks the model
    to judge first whether the supplied facts suffice ("sufficient" field) and
    to reply 证据不足 otherwise. The baselines can therefore decline in their
    own words; `abstained` records that verdict. (An earlier docstring claimed
    that none of them can abstain; that was no longer true and is removed.)
  * KAPING retrieves facts by similarity to the question (sentence embeddings
    when sentence-transformers and the model are available, otherwise
    character-overlap ranking) and injects the top-k. The retriever actually
    used is written to every record (`retriever`). The earlier version injected
    the first 40 facts of the 12 most frequent relations without any ranking,
    which is not KAPING.
  * They are re-implementations against this graph schema, not the authors'
    code. The graph has 365 relation types and property-value facts; published
    ToG assumes Freebase/Wikidata-style triples. The paper's baseline section
    must say "adapted", not "as published".
  * Search budget is identical to SCEPS where the method has the concept:
    depth 3, beam 3, candidate pool 8, max 12 candidates per relation.
  * Entity resolution is shared with SCEPS (KG_QA/search.extract_topic_entities,
    including the entity guard), so differences are not due to resolution.

Usage:
    python tools/run_baselines.py --tag base-ds --methods kaping,tog,tog2,search_on_graph \\
        --model deepseek-flash --api-base https://api.deepseek.com/v1 --api-key ... \\
        --concurrency 200 --neo4j-password ...
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
sys.path.insert(0, str(PROJECT_DIR / "KG_QA"))

from graph import HazardNeo4jClient  # noqa: E402
from search import (  # noqa: E402
    extract_query_slots,
    extract_topic_entities,
    _compute_path_score,
    _serialize_path_record,
)
from utils import (  # noqa: E402
    call_chat_json,
    configure_client,
    configure_http_pool,
    load_qa_csv,
    overlap_score,
    track_llm_usage,
)

GRAPH_METHODS = ("kaping", "tog", "tog2", "search_on_graph")
LLM_ONLY_METHODS = ("zero_shot", "few_shot", "cot")
METHODS = GRAPH_METHODS + LLM_ONLY_METHODS
DEFAULT_EMBED_MODEL = os.getenv("KGQA_EMBED_MODEL", "shibing624/text2vec-base-chinese")

# Built from chemicals that occur in none of the 700 test questions/answers
# (checked against Question&Answer/*.csv on 2026-11-17); answers are the graph
# values. One example per common question type, including one two-hop example.
FEW_SHOT_EXAMPLES = (
    ("(正)丁酸乙烯酯的闪点是多少？", "12℃"),
    ("1,2-二溴丙烷的禁配物是什么？", "强碱、强氧化剂、镁"),
    ("1,2-二甲基环戊烷发生火灾时，可选用哪种灭火剂？", "泡沫、二氧化碳、干粉"),
    ("1,5-环辛二烯被划为易燃液体,类别3的危险性类别后，应采取什么预防措施？",
     "远离热源/火花/明火/热表面———禁止吸烟。；保持容器密闭。"),
)

LLM_ONLY_SYSTEM = (
    "你是危化品安全问答助手。请依据你掌握的知识回答问题；"
    "如果不知道或无法确定，就直接说明无法回答。输出必须是一个 JSON 对象。"
)

# --------------------------------------------------------------------- prompts ---

ANSWER_SYSTEM = (
    "你是危化品安全问答助手。你必须严格依据给出的图谱事实回答问题。"
    "如果给出的证据不足以回答，就直接说明证据不足。"
    "输出必须是一个 JSON 对象。"
)

KAPING_SYSTEM = (
    "你是危化品安全问答助手。下面会给出从知识图谱中检索到的若干事实。"
    "请仅使用这些事实回答问题；事实未覆盖的内容不要补充。"
    "输出必须是一个 JSON 对象。"
)

TOG_RELATION_SYSTEM = (
    "你在知识图谱上做逐步推理。给定当前节点可用的关系列表，"
    "选出最有可能通向答案的关系。输出必须是一个 JSON 对象。"
)

SOG_OBSERVE_SYSTEM = (
    "你在知识图谱上做逐步导航。你会看到当前节点的一个关系，"
    "以及沿该关系得到的所有候选。请判断下一步怎么走。"
    "输出必须是一个 JSON 对象。"
)


def _render_facts(facts: list[str]) -> str:
    if not facts:
        return "（无）"
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))


def _answer_prompt(question: str, facts: list[str], system_note: str = "") -> str:
    return (
        f"问题：{question}\n"
        f"{system_note}"
        f"图谱事实：\n{_render_facts(facts)}\n"
        "先判断这些事实是否足以回答问题，再作答。\n"
        '输出格式：{"sufficient": true 或 false, "answer":"...", "evidence":"..."}\n'
        "sufficient=false 表示给出的图谱事实不足以回答，此时 answer 写「证据不足，无法回答。」\n"
        "sufficient=true 表示事实足以回答，此时 answer 必须直接给出答案。"
    )


# ------------------------------------------------------------------- shared ---

def resolve_entities(question: str, args, graph_client: HazardNeo4jClient,
                     query_slots: dict[str, Any]) -> list[str]:
    return extract_topic_entities(question, graph_client, args, query_slots=query_slots)


def node_relations(node_id: str, graph_client: HazardNeo4jClient) -> list[dict[str, Any]]:
    return graph_client.get_available_descriptors(node_id)


def expand(node_id: str, relation: str, graph_client: HazardNeo4jClient,
           limit: int) -> list[dict[str, Any]]:
    return graph_client.expand_descriptor(node_id, relation, limit)


def rank_relations(question: str, node_name: str, relations: list[dict[str, Any]],
                   args, width: int, *, system_prompt: str = TOG_RELATION_SYSTEM,
                   path_summary: str = "") -> list[tuple[str, float]]:
    """One LLM call: score the available relations of a node."""
    if not relations:
        return []
    lines = "\n".join(f"{i + 1}. {r['relation']}" for i, r in enumerate(relations))
    payload = call_chat_json(
        api_base=args.api_base,
        api_key=args.opeani_api_keys,
        model=args.LLM_type,
        system_prompt=system_prompt,
        user_prompt=(
            f"选出最多 {width} 个最值得继续扩展的关系/属性。\n"
            '输出 JSON：{"relations":[{"name":"...","score":0.95}]}\n'
            f"问题：{question}\n"
            f"当前节点：{node_name}\n"
            f"已走路径：{path_summary or '无'}\n"
            f"候选关系：\n{lines}"
        ),
        temperature=0.0,
        max_tokens=args.max_length,
        reasoning="none",
    )
    if not isinstance(payload, dict):
        return [(r["relation"], 0.5) for r in relations[:width]]
    scored = []
    for item in payload.get("relations") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if name:
            scored.append((name, score))
    if not scored:
        return [(r["relation"], 0.5) for r in relations[:width]]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:width]


def pick_candidates(question: str, relation: str, candidates: list[dict[str, Any]],
                    args, top_k: int) -> list[dict[str, Any]]:
    """One LLM call: score returned candidate values for a relation."""
    if not candidates:
        return []
    lines = "\n".join(
        f"{i + 1}. {str(c.get('name') or '').strip()}"
        for i, c in enumerate(candidates)
    )
    payload = call_chat_json(
        api_base=args.api_base,
        api_key=args.opeani_api_keys,
        model=args.LLM_type,
        system_prompt=(
            "你是危化品问答中的证据候选筛选器。从候选值中挑出最能回答问题的一项或几项。"
            "输出必须是一个 JSON 对象。"
        ),
        user_prompt=(
            f"选出最多 {top_k} 个最相关的候选。\n"
            '输出 JSON：{"keep":[{"index":1,"score":0.9}]}\n'
            f"问题：{question}\n"
            f"关系：{relation}\n"
            f"候选：\n{lines}"
        ),
        temperature=0.0,
        max_tokens=args.max_length,
        reasoning="none",
    )
    if not isinstance(payload, dict):
        return candidates[:top_k]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in payload.get("keep") or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index")) - 1
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            scored.append((score, candidates[idx]))
    if not scored:
        return candidates[:top_k]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def generate(question: str, facts: list[str], args, *,
             system_prompt: str = ANSWER_SYSTEM,
             system_note: str = "") -> tuple[str, str, bool]:
    """One generation call. Returns (answer, evidence, abstained).

    The baselines have no structural abstention path, so abstention is measured
    from the model's own `sufficient` verdict rather than assumed. An earlier
    version hardcoded abstained=False for every baseline, which turned
    "answered" into a constant 100% and made the comparison meaningless.
    """
    payload = call_chat_json(
        api_base=args.api_base,
        api_key=args.opeani_api_keys,
        model=args.LLM_type,
        system_prompt=system_prompt,
        user_prompt=_answer_prompt(question, facts, system_note),
        temperature=0.0,
        max_tokens=args.max_length,
        reasoning="auto",
    )
    if not isinstance(payload, dict):
        return "证据不足，无法回答。", "", True
    answer = str(payload.get("answer", "") or "").strip()
    evidence = str(payload.get("evidence", "") or "").strip()
    sufficient = payload.get("sufficient")
    if isinstance(sufficient, bool):
        return answer, evidence, not sufficient
    # Older/odd outputs: fall back to reading the text. Kept explicit so the
    # fallback is visible in the record rather than silently guessing.
    decline_markers = ("证据不足", "无法回答", "无法确定", "未提供", "不足以")
    return answer, evidence, any(m in answer for m in decline_markers)


# ------------------------------------------------------------------ baselines ---

_EMBEDDER: Any = None          # None = not loaded yet, False = unavailable
_EMBEDDER_NAME = ""
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder(model_name: str):
    """Load the sentence-embedding model once; return None if unavailable."""
    global _EMBEDDER, _EMBEDDER_NAME
    if not model_name or model_name == "lexical":
        return None
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            try:
                # The installed brotli build rejects the output_buffer_limit
                # keyword that httpx2 passes during decompression, which breaks
                # every huggingface_hub download. The shim swaps in a wrapper
                # that accepts it; without it the model cannot be fetched.
                # Loaded by path because this file runs as a script, so `tools`
                # is not an importable package here.
                import importlib.util as _ilu
                _shim_path = Path(__file__).resolve().parent / "brotli_compat.py"
                if _shim_path.is_file():
                    _spec = _ilu.spec_from_file_location("_brotli_compat", _shim_path)
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
                from sentence_transformers import SentenceTransformer
                _EMBEDDER = SentenceTransformer(model_name)
                _EMBEDDER_NAME = model_name
            except Exception as exc:  # missing package, no network, bad name
                _EMBEDDER = False
                print(f"[kaping] embedding model '{model_name}' unavailable "
                      f"({type(exc).__name__}: {exc}); using lexical ranking. "
                      f"This is recorded per question in the `retriever` field.")
        return _EMBEDDER or None


def rank_facts(question: str, facts: list[str], args) -> tuple[list[str], str]:
    """Order verbalised facts by similarity to the question (KAPING retrieval)."""
    embedder = _get_embedder(getattr(args, "kaping_embed_model", DEFAULT_EMBED_MODEL))
    if embedder is not None:
        with _EMBEDDER_LOCK:  # encode() is not guaranteed thread-safe
            q_vec = embedder.encode([question], normalize_embeddings=True)
            f_vec = embedder.encode(facts, normalize_embeddings=True, batch_size=64)
        scores = f_vec @ q_vec[0]
        order = sorted(range(len(facts)), key=lambda i: float(scores[i]), reverse=True)
        return [facts[i] for i in order], f"embedding:{_EMBEDDER_NAME}"
    order = sorted(range(len(facts)), key=lambda i: overlap_score(question, facts[i]), reverse=True)
    return [facts[i] for i in order], "lexical_overlap"


def run_kaping(question: str, entities: list[str], graph_client: HazardNeo4jClient,
               args) -> dict[str, Any]:
    """KAPING: retrieve the facts most similar to the question and inject the top-k.

    All one-hop facts (relations and properties) of the resolved chemicals are
    verbalised, ranked by similarity to the question, and the top
    ``args.kaping_top_k`` go into a single generation prompt. No iteration.
    """
    facts: list[str] = []
    for entity in entities:
        node = graph_client.get_chemical_node(entity)
        if not node:
            continue
        node_id = node["element_id"]
        for rel in node_relations(node_id, graph_client):
            name = rel["relation"]
            for cand in expand(node_id, name, graph_client, args.max_candidates_per_relation):
                value = str(cand.get("name") or "").strip()
                if value:
                    if name.startswith("属性:"):
                        facts.append(f"{entity} 的 {name.split(':', 1)[1]} 是 {value}")
                    else:
                        facts.append(f"{entity} -[{name}]-> {value}")
    if not facts:
        answer, evidence, abstained = generate(question, [], args, system_prompt=KAPING_SYSTEM)
        return {"answer": answer, "evidence": evidence, "paths": [], "sampled_facts": [],
                "abstained": abstained, "retriever": "none", "n_candidate_facts": 0}

    ranked, retriever = rank_facts(question, facts, args)
    top = ranked[: max(1, int(getattr(args, "kaping_top_k", 10)))]
    answer, evidence, abstained = generate(
        question, top, args, system_prompt=KAPING_SYSTEM,
        system_note="以下事实是按与问题的相似度从知识图谱中检索出的前若干条。\n",
    )
    return {"answer": answer, "evidence": evidence, "paths": [], "sampled_facts": top,
            "abstained": abstained, "retriever": retriever, "n_candidate_facts": len(facts)}


def run_llm_only(question: str, args, *, style: str) -> dict[str, Any]:
    """Zero-shot, few-shot, or chain-of-thought prompting without the graph."""
    fmt = '输出格式：{"answer":"..."}'
    if style == "zero_shot":
        prompt = f"问题：{question}\n{fmt}"
    elif style == "few_shot":
        shots = "\n".join(f"问题：{q}\n答案：{a}" for q, a in FEW_SHOT_EXAMPLES)
        prompt = f"下面是几个示例：\n{shots}\n\n问题：{question}\n{fmt}"
    elif style == "cot":
        prompt = (f"问题：{question}\n请先逐步推理，再给出最终答案。\n"
                  '输出格式：{"reasoning":"...","answer":"..."}')
    else:
        raise ValueError(style)
    payload = call_chat_json(
        api_base=args.api_base,
        api_key=args.opeani_api_keys,
        model=args.LLM_type,
        system_prompt=LLM_ONLY_SYSTEM,
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=max(int(args.max_length), 2048),
        reasoning="auto",
    )
    if not isinstance(payload, dict):
        return {"answer": "无法回答。", "evidence": "", "paths": [], "sampled_facts": [], "abstained": True}
    answer = str(payload.get("answer", "") or "").strip()
    decline_markers = ("无法回答", "无法确定", "不知道", "不清楚", "证据不足")
    return {"answer": answer, "evidence": str(payload.get("reasoning", "") or "")[:2000], "paths": [],
            "sampled_facts": [], "abstained": (not answer) or any(m in answer for m in decline_markers)}


def run_tog(question: str, entities: list[str], graph_client: HazardNeo4jClient,
            args, *, knowledge_guided: bool) -> dict[str, Any]:
    """Iterative beam search over the graph.

    Distinct from SCEPS in the acceptance rule only: ToG accepts the best-scoring
    paths at the depth limit and never asks whether the evidence is complete.
    That is exactly the property the paper's ablation separates, so sharing the
    search mechanics keeps the comparison about acceptance, not about
    implementation quality.
    """
    beam: list[dict[str, Any]] = []
    for entity in entities:
        node = graph_client.get_chemical_node(entity)
        if not node:
            continue
        beam.append({
            "node_id": node["element_id"],
            "node_name": entity,
            "triplets": [],
            "score": 1.0,
        })

    for depth in range(1, args.depth + 1):
        next_beam: list[dict[str, Any]] = []
        for path in beam[: args.width]:
            relations = node_relations(path["node_id"], graph_client)
            ranked = rank_relations(
                question, path["node_name"], relations, args, args.width,
                path_summary=" | ".join(path["triplets"]),
            )
            for relation, rel_score in ranked:
                candidates = expand(path["node_id"], relation, graph_client,
                                    args.max_candidates_per_relation)
                selected = pick_candidates(question, relation, candidates, args, args.width)
                for cand in selected:
                    value = str(cand.get("name") or "").strip()
                    if not value:
                        continue
                    if relation.startswith("属性:"):
                        triple = f"{path['node_name']} 的 {relation.split(':', 1)[1]} 是 {value}"
                    else:
                        triple = f"{path['node_name']} -[{relation}]-> {value}"
                    step = _compute_path_score([{
                        "relation_score": rel_score,
                        "candidate_score": 0.8,
                    }])
                    new_score = (path["score"] * step) ** 0.5 if path["score"] < 1.0 else step
                    next_beam.append({
                        "node_id": cand.get("element_id"),
                        "node_name": value,
                        "triplets": path["triplets"] + [triple],
                        "score": new_score,
                    })
        if not next_beam:
            break
        next_beam.sort(key=lambda p: p["score"], reverse=True)
        beam = next_beam[: args.width]

    facts = [" | ".join(p["triplets"]) for p in beam[: args.top_k_paths_for_reasoning]]
    if knowledge_guided:
        note = (
            "下面每条事实是一条推理链，链上的关系按顺序连接。"
            "请沿着链的顺序组织答案，并说明中间经过的节点。\n"
        )
        answer, evidence, abstained = generate(question, facts, args, system_note=note)
    else:
        flat = [t for p in beam[: args.top_k_paths_for_reasoning] for t in p["triplets"]]
        answer, evidence, abstained = generate(question, flat[:30], args)
    return {
        "answer": answer,
        "evidence": evidence,
        "paths": [p["triplets"] for p in beam[: args.top_k_paths_for_reasoning]],
        "sampled_facts": facts,
        "abstained": abstained,
    }


def run_search_on_graph(question: str, entities: list[str],
                        graph_client: HazardNeo4jClient, args) -> dict[str, Any]:
    """Observe-then-navigate with a single trajectory.

    Each step shows the model one relation's candidates and asks it to either
    descend or stop and answer. Stopping is what makes this method answer
    single-hop questions well; having no completeness check is what the paper
    contrasts against.
    """
    if not entities:
        return {"answer": "未识别到化学品实体。", "evidence": "", "paths": [],
                "sampled_facts": []}
    entity = entities[0]
    node = graph_client.get_chemical_node(entity)
    if not node:
        return {"answer": "未识别到化学品实体。", "evidence": "", "paths": [],
                "sampled_facts": []}

    node_id, node_name = node["element_id"], entity
    triples: list[str] = []

    for step in range(1, args.depth + 1):
        relations = node_relations(node_id, graph_client)
        if not relations:
            break
        ranked = rank_relations(
            question, node_name, relations, args, 1,
            system_prompt=SOG_OBSERVE_SYSTEM,
            path_summary=" | ".join(triples),
        )
        if not ranked:
            break
        relation, _ = ranked[0]
        candidates = expand(node_id, relation, graph_client,
                            args.max_candidates_per_relation)
        selected = pick_candidates(question, relation, candidates, args, 1)
        if not selected:
            break
        cand = selected[0]
        value = str(cand.get("name") or "").strip()
        if not value:
            break
        if relation.startswith("属性:"):
            triples.append(f"{node_name} 的 {relation.split(':', 1)[1]} 是 {value}")
        else:
            triples.append(f"{node_name} -[{relation}]-> {value}")

        # Ask whether to continue or answer now. This is the "observe" half of
        # the paradigm: the decision is the model's, not a slot check.
        decision = call_chat_json(
            api_base=args.api_base,
            api_key=args.opeani_api_keys,
            model=args.LLM_type,
            system_prompt=SOG_OBSERVE_SYSTEM,
            user_prompt=(
                f'输出 JSON：{{"action":"answer" 或 "continue"}}\n'
                f"问题：{question}\n"
                f"已收集证据：\n{_render_facts(triples)}\n"
                "这些证据是否已足够回答问题？"
            ),
            temperature=0.0,
            max_tokens=128,
            reasoning="none",
        )
        action = str((decision or {}).get("action", "")).strip().lower()
        if action == "answer":
            break
        if not cand.get("element_id"):
            break
        node_id, node_name = cand["element_id"], value

    answer, evidence, abstained = generate(question, triples, args)
    return {"answer": answer, "evidence": evidence,
            "paths": [triples] if triples else [], "sampled_facts": triples,
            "abstained": abstained}


# ------------------------------------------------------------------- driver ---

def run_one(qid: int, question: str, gold: str, method: str,
            args: SimpleNamespace, graph_client: HazardNeo4jClient) -> dict[str, Any]:
    started = time.time()
    query_slots: dict[str, Any] = {}
    entities: list[str] = []
    with track_llm_usage() as usage:
        if method in LLM_ONLY_METHODS:
            # No graph access and no slot parsing: the question goes to the model as is.
            out = run_llm_only(question, args, style=method)
        else:
            query_slots = extract_query_slots(question, args)
            entities = resolve_entities(question, args, graph_client, query_slots)
            if method == "kaping":
                out = run_kaping(question, entities, graph_client, args)
            elif method == "tog":
                out = run_tog(question, entities, graph_client, args, knowledge_guided=False)
            elif method == "tog2":
                out = run_tog(question, entities, graph_client, args, knowledge_guided=True)
            elif method == "search_on_graph":
                out = run_search_on_graph(question, entities, graph_client, args)
            else:
                raise ValueError(f"unknown method: {method}")

    return {
        "qid": qid,
        "question": question,
        "gold": gold,
        "method": method,
        "options": {
            "model": args.LLM_type,
            "entity_guard": getattr(args, "entity_guard", "on"),
            "kaping_top_k": getattr(args, "kaping_top_k", None) if method == "kaping" else None,
            "retriever": out.get("retriever") if method == "kaping" else None,
            "n_candidate_facts": out.get("n_candidate_facts") if method == "kaping" else None,
        },
        "slots": {k: v for k, v in query_slots.items() if not str(k).startswith("_")},
        "resolution": query_slots.get("_resolution", {}),
        "entities": entities,
        "answer": out["answer"],
        "evidence": out["evidence"],
        "reasoning_paths": [" | ".join(p) for p in out["paths"]],
        "sampled_facts": out.get("sampled_facts", [])[:40],
        # The model's own "sufficient"/decline verdict (see generate()); there
        # is no structural abstention path in the baselines.
        "abstained": bool(out.get("abstained", False)),
        "forced_generation": False,
        "degraded_no_llm": usage.request_count == 0,
        "entity_packets": [{
            "entity": e,
            "status": "answered",
            "triplets": [t for p in out["paths"] for t in p] if out["paths"] else [],
            "answered_paths": [{"triplets": p, "path_score": 1.0} for p in out["paths"]],
            "dead_paths": [],
        } for e in entities] if entities else [],
        "usage": usage.to_dict(),
        "time_cost": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline matrix.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--splits", default="single_hop,multi_hop,hallucination")
    parser.add_argument("--input-csv", default=str(PROJECT_DIR / "Question&Answer"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--top-k-paths", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--max-candidates-per-relation", type=int, default=12)
    parser.add_argument("--kaping-top-k", type=int, default=10,
                        help="number of retrieved facts injected by KAPING")
    parser.add_argument("--kaping-embed-model", default=DEFAULT_EMBED_MODEL,
                        help="sentence-transformers model for KAPING retrieval; 'lexical' forces "
                             "character-overlap ranking")
    parser.add_argument("--entity-guard", default="on", choices=["on", "off"],
                        help="must match the SCEPS run being compared against")
    parser.add_argument("--requests-per-minute", type=int, default=0)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE",
                                                        "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", ""))
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()

    methods = [m.strip() for m in ns.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown methods: {unknown}")
    if not ns.api_key:
        sys.exit("need --api-key or DEEPSEEK_API_KEY")
    needs_graph = any(m in GRAPH_METHODS for m in methods)
    if needs_graph and not ns.neo4j_password:
        sys.exit("need --neo4j-password or KGQA_NEO4J_PASSWORD")

    args = SimpleNamespace(
        max_length=ns.max_length,
        temperature_exploration=0.0,
        temperature_reasoning=0.0,
        width=ns.width,
        depth=ns.depth,
        min_relation_score=0.3,
        min_candidate_score=0.3,
        top_k_paths_for_reasoning=ns.top_k_paths,
        candidate_pool_size=ns.candidate_pool,
        max_candidates_per_relation=ns.max_candidates_per_relation,
        LLM_type=ns.model,
        entity_model=ns.model,
        api_base=ns.api_base,
        opeani_api_keys=ns.api_key,
        ablation="full",
        entity_guard=ns.entity_guard,
        kaping_top_k=ns.kaping_top_k,
        kaping_embed_model=ns.kaping_embed_model,
    )

    if ns.dry_run:
        print(f"baseline sweep: {ns.tag}")
        for method in methods:
            for split in ns.splits.split(","):
                path = Path(ns.output_dir) / ns.tag / method / f"{split}.jsonl"
                n = sum(1 for line in path.open(encoding="utf-8")) if path.is_file() else 0
                print(f"  {method:18s} {split:14s} done={n}")
        print("\n(dry run: nothing executed)")
        return

    configure_client([ns.api_key], ns.requests_per_minute)
    configure_http_pool(ns.concurrency + 16)
    graph_client = (
        HazardNeo4jClient(ns.neo4j_uri, ns.neo4j_user, ns.neo4j_password,
                          max_pool_size=ns.concurrency + 20)
        if needs_graph else None
    )
    write_lock = threading.Lock()

    try:
        for method in methods:
            for split in [s.strip() for s in ns.splits.split(",") if s.strip()]:
                csv_path = Path(ns.input_csv) / f"{split}.csv"
                if not csv_path.is_file():
                    print(f"skip {split}: not found")
                    continue
                rows = load_qa_csv(csv_path)
                out_dir = Path(ns.output_dir) / ns.tag / method
                out_dir.mkdir(parents=True, exist_ok=True)
                jsonl = out_dir / f"{split}.jsonl"
                done = set()
                if jsonl.is_file():
                    with jsonl.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    done.add(int(json.loads(line)["qid"]))
                                except (json.JSONDecodeError, KeyError, ValueError):
                                    pass
                todo = [(i, r["question"], r.get("reference_answer", ""))
                        for i, r in enumerate(rows, start=1) if i not in done]
                if ns.limit:
                    todo = todo[: ns.limit]
                if not todo:
                    print(f"{method}/{split}: nothing to do ({len(done)} done)")
                    continue

                print(f"\n=== {method}/{split}: {len(todo)} questions, "
                      f"concurrency {ns.concurrency} ===")
                started = time.time()
                completed = 0
                with jsonl.open("a", encoding="utf-8") as fh:
                    with ThreadPoolExecutor(max_workers=ns.concurrency) as pool:
                        futures = {
                            pool.submit(run_one, qid, q, g, method, args, graph_client): qid
                            for qid, q, g in todo
                        }
                        for fut in as_completed(futures):
                            qid = futures[fut]
                            completed += 1
                            try:
                                rec = fut.result()
                            except Exception as exc:
                                rec = {"qid": qid,
                                       "error": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()[-800:]}
                            with write_lock:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fh.flush()
                            if completed % 25 == 0 or completed == len(todo):
                                el = time.time() - started
                                print(f"  {completed}/{len(todo)}  "
                                      f"{completed/el:.2f} q/s  {el/60:.1f}m")
                print(f"  wrote {jsonl}")
    finally:
        if graph_client is not None:
            graph_client.close()


if __name__ == "__main__":
    main()
