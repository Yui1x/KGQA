from __future__ import annotations

import json
from typing import Any

from utils import render_path_chain

ENTITY_EXTRACTION_SYSTEM = (
    "你是危化品问答中的槽位化问题解析器。"
    "你需要从问题文本本身抽取化学品，并识别多跳问题相关槽位。"
    "不要补全题目中未出现的化学品，不要解释。"
    "输出必须是一个JSON对象。"
)


def build_entity_extraction_prompt(question: str) -> str:
    # v2 (2026-11-17 revision): `constraint` now means the conditions that
    # change which graph fact applies (concentration, physical form,
    # formulation, item, exposure route), matching the paper's definition of
    # K_q. The previous wording asked for degree words such as 最早/一般, which
    # is why removing the constraint slot had no measurable effect. The bridge
    # vs target distinction is also spelled out, because the parser used to
    # put the asked field into `bridge` for questions like
    # "X的一个燃烧产物是什么", leaving `target` empty.
    return (
        "请对下面问题做槽位化解析。\n"
        "要求：\n"
        "1. chemicals: 只抽取问题中直接出现的化学品，数量不固定，可以是0个、1个或多个。"
        "必须保留题目中的完整写法，包括括号、位次词以及浓度、形态、配方等修饰，"
        "例如 (对)异丙基苯酚、含量≥60%的双氧水、乙酸溶液、无涂层的铝 都要原样保留。\n"
        "2. event: 问题场景事件，例如 起火、泄漏、皮肤接触、眼睛接触、吸入、运输；没有就写空字符串。\n"
        "3. bridge: 只有当问题需要先找到一个中间对象、再问这个中间对象的信息时才填写。"
        "例如 'X的燃烧产物的急救措施' 中 bridge=燃烧产物；'X被划为某危险性类别后的防护要求' 中 bridge=危险性类别；"
        "'X的禁配物泄漏时怎么处置' 中 bridge=禁配物。"
        "如果问题直接问某个字段（例如 'X的燃烧产物是什么'、'X的禁配物是什么'），这个字段是 target，bridge 留空。\n"
        "4. target: 最终要回答的字段，尽量用图谱字段式的短语，例如 燃烧产物、禁配物、推荐灭火剂、急救措施、"
        "防护要求、分类依据、危险说明、闪点、外观与性状；问题总有要回答的内容，不要留空。\n"
        "5. constraint: 会改变适用哪条图谱事实的限定条件列表，例如 浓度或含量（30%、含量≥60%）、"
        "物理形态（固体、粉末、溶液、液化、气体）、配方或规格（母液、水合物、无水、有涂层、无涂层、工业级）、"
        "物品或包装（钢瓶）、暴露途径（经皮、吸入、经口）；没有就返回空列表。"
        "不要把 '一般'、'最早'、'严重'、'立即' 这类程度词当作约束。\n"
        "6. 不要添加题目中没有出现的实体。\n"
        '7. 只输出JSON，例如 {"chemicals":["含量≥60%的双氧水"],"event":"","bridge":"危险性类别",'
        '"target":"分类依据","constraint":["含量≥60%"]}。\n'
        f"问题：{question}"
    )


def _format_query_slots(query_slots: dict[str, Any] | None) -> str:
    if not query_slots:
        return ""
    event = str(query_slots.get("event", "") or "").strip() or "无"
    bridge = str(query_slots.get("bridge", "") or "").strip() or "无"
    target = str(query_slots.get("target", "") or "").strip() or "无"
    constraint_value = query_slots.get("constraint", [])
    if isinstance(constraint_value, list):
        constraint = ", ".join(str(item).strip() for item in constraint_value if str(item).strip()) or "无"
    else:
        constraint = str(constraint_value or "").strip() or "无"
    return (
        f"Event: {event}\n"
        f"Bridge: {bridge}\n"
        f"Target: {target}\n"
        f"Constraint: {constraint}\n"
    )


RELATION_PRUNE_SYSTEM = (
    "你是危化品问答中的关系/属性裁剪器。"
    "你需要从候选关系和候选属性中选出当前最值得探索的少量项。"
    "输出必须是一个JSON对象。"
)


def build_relation_prune_prompt(
    *,
    question: str,
    root_entity: str,
    current_node: dict[str, Any],
    path_summary: str,
    clue: str,
    relation_rows: list[dict[str, Any]],
    width: int,
    query_slots: dict[str, Any] | None = None,
) -> str:
    relation_lines = [
        f"{idx + 1}. {item['relation']}"
        for idx, item in enumerate(relation_rows)
    ]
    current_name = current_node.get("name") or root_entity
    slot_block = _format_query_slots(query_slots)
    return (
        f"选出最多 {width} 个最值得继续扩展的关系/属性。\n"
        "打分参考：0.9-1.0 直接且高度相关；0.7-0.89 强相关；0.4-0.69 部分相关；0.0-0.39 弱相关或噪音。\n"
        "请尽量拉开分数差距，不要把大部分候选都打成接近分。\n"
        '输出必须是 JSON，优先使用 {"relations":[{"name":"...","score":0.95}, ...]} 这种格式。\n'
        "每个候选项必须包含 name 和 score 两个字段，不要改字段名。\n"
        f"Q: {question}\n"
        f"Root: {root_entity}\n"
        f"Node: {current_name}\n"
        f"Path: {path_summary or '无'}\n"
        f"Clue: {clue or '无'}\n"
        f"{slot_block}"
        "Candidates:\n"
        + "\n".join(relation_lines)
    )


CANDIDATE_PRUNE_SYSTEM = (
    "你是危化品问答中的实体/证据候选打分器。"
    "你需要从候选扩展项中保留最能帮助回答问题的少量项。"
    "输出必须是一个JSON对象。"
)


def build_candidate_prune_prompt(
    *,
    question: str,
    root_entity: str,
    relation_name: str,
    path_summary: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    query_slots: dict[str, Any] | None = None,
) -> str:
    lines = []
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. {candidate['name']} | {candidate['triplet']}"
        )
    slot_block = _format_query_slots(query_slots)
    return (
        f"从下列候选扩展项中保留最多 {top_k} 个最有帮助的项。\n"
        "打分参考：0.9-1.0 直接回答问题核心；0.7-0.89 强相关；0.4-0.69 只有部分帮助；0.0-0.39 基本无帮助。\n"
        "请必须拉开最优项和次优项的分数差距。\n"
        '输出必须是 JSON，优先使用 {"selected":[{"name":"...","score":0.95}, ...]} 这种格式。\n'
        "name 字段只写候选值本身，不要带上 | 之后的三元组部分。\n"
        "每个候选项必须包含 name 和 score 两个字段，不要改字段名。\n"
        f"Q: {question}\n"
        f"Root: {root_entity}\n"
        f"Relation: {relation_name}\n"
        f"Path: {path_summary or '无'}\n"
        f"{slot_block}"
        "Candidates:\n"
        + "\n".join(lines)
    )


ENTITY_REASONING_SYSTEM = (
    "你是危化品问答中的单实体停止判定器。"
    "你只判断当前单个实体是否已经找到符合问题要求的路径。"
    "输出必须是一个JSON对象。"
)


def build_entity_reasoning_prompt(
    *,
    question: str,
    entity_name: str,
    paths: list[str],
    clue: str,
    query_slots: dict[str, Any] | None = None,
) -> str:
    prompt = {
        "sufficient": True,
        "missing": "",
    }
    current_path = paths[0] if paths else "无"
    slot_block = _format_query_slots(query_slots)
    return (
        "请只针对当前实体的当前单条路径做判断。\n"
        "你的任务是判断这一条路径本身是否已经足够回答问题中与该实体相关的部分，是否已经覆盖全部槽位。\n"
        "只有当这一条路径本身已经能支撑作答时，才返回 sufficient=true，否则返回 sufficient=false。\n"
        f"输出格式必须为 JSON，例如：{json.dumps(prompt, ensure_ascii=False)}\n"
        f"问题：{question}\n"
        f"当前实体：{entity_name}\n"
        f"当前路径：{current_path}\n"
        f"当前缺失线索：{clue or '无'}\n"
        f"{slot_block}"
    )


JOINT_SUFFICIENCY_SYSTEM = (
    "你是危化品知识图谱问答中的证据充分性判定器。"
    "你只依据给出的候选证据路径做判断，不使用路径之外的知识。"
    "输出必须是一个JSON对象。"
)


def build_joint_sufficiency_prompt(
    *,
    question: str,
    entity_name: str,
    paths: list[str],
    clue: str,
    query_slots: dict[str, Any] | None = None,
    min_hops: int = 1,
) -> str:
    """Judge a *set* of candidate paths for one chemical.

    The per-path judge (build_entity_reasoning_prompt) can never accept an
    answer that is spread over several sibling facts, e.g. a precaution list
    whose items are separate graph tails. Error analysis showed this is where
    most multi-hop abstentions came from (gold answers with 2-3 parts). Here
    the judge sees the candidate pool and names the supporting subset, which
    the caller then validates deterministically.
    """
    numbered = "\n".join(f"{idx}. {path}" for idx, path in enumerate(paths, start=1)) or "（无）"
    structural = (
        "本题需要经过中间环节（Bridge）：支持路径中至少要有一条经过该中间环节、不少于2跳的路径。\n"
        if min_hops >= 2 else ""
    )
    example = {
        "sufficient": True,
        "support": [1, 3],
        "slot_support": {"target": [1, 3], "bridge": [1], "constraint": []},
        "missing": "",
    }
    return (
        "下面是同一个化学品在知识图谱中的若干条候选证据路径。"
        "请判断其中是否存在一组路径，合起来足以回答问题中与该化学品相关的部分。\n"
        "判定规则：\n"
        "1. 必须有路径给出问题所问目标字段（Target）的内容。\n"
        "2. 如果有中间环节（Bridge），支持路径必须经过它，而且目标字段必须属于中间对象，"
        "不能用化学品自身的同名字段代替。\n"
        "3. 如果有约束（Constraint，如浓度、形态、配方、有无涂层、物品类型、暴露途径），"
        "必须有路径中的关系名或实体明确对应该约束；不能用不带该约束的一般信息代替。\n"
        "4. 问题同时问多个方面时（例如同时问隔离距离和灭火剂），每个方面都要有路径支撑；"
        "可以组合多条路径。\n"
        "5. 只能依据路径中的事实。\n"
        f"{structural}"
        f"输出JSON，例如：{json.dumps(example, ensure_ascii=False)}\n"
        "support 填支持路径的编号；sufficient=true 时 support 不能为空；"
        "sufficient=false 时在 missing 中说明缺什么。\n"
        f"问题：{question}\n"
        f"当前化学品：{entity_name}\n"
        f"当前缺失线索：{clue or '无'}\n"
        f"{_format_query_slots(query_slots)}"
        "候选路径：\n"
        f"{numbered}"
    )


SLOT_REPAIR_SYSTEM = (
    "你是危化品问答中的槽位修正器。"
    "上一轮检索没有找到充分证据，你需要根据图谱中实际可用的字段修正问题的槽位解析。"
    "输出必须是一个JSON对象。"
)


def build_slot_repair_prompt(
    *,
    question: str,
    entity_name: str,
    query_slots: dict[str, Any] | None,
    missing: str,
    available_fields: list[str],
) -> str:
    fields = "、".join(available_fields[:80]) or "（无）"
    current = json.dumps(
        {k: (query_slots or {}).get(k, "" if k != "constraint" else []) for k in ("event", "bridge", "target", "constraint")},
        ensure_ascii=False,
    )
    return (
        "上一轮按下面的槽位检索后，判定器认为证据不充分。请检查槽位解析是否有误并给出修正。\n"
        "常见错误：把直接询问的字段误放进 bridge；target 与图谱字段名不一致；漏掉或误加中间环节；"
        "把程度词当成约束。\n"
        "可以把 target、bridge 改写为图谱中实际存在的字段名；不要改动问题中出现的约束。\n"
        '输出JSON：{"event":"...","bridge":"...","target":"...","constraint":[...],"changed":true或false}\n'
        f"问题：{question}\n"
        f"化学品：{entity_name}\n"
        f"当前槽位：{current}\n"
        f"判定器指出的缺失：{missing or '无'}\n"
        f"该化学品在图谱中可用的字段：{fields}"
    )


FINAL_AGGREGATION_SYSTEM = (
    "你是危化品问答中的最终答案生成器。"
    "你会收到一个或多个实体分别检索完成后的证据包。"
    "你的任务是忠实地基于这些证据回答原问题。"
    "输出必须是一个JSON对象。"
)


def build_final_aggregation_prompt(question: str, entity_packets: list[dict[str, Any]]) -> str:
    packet_lines = []
    for packet in entity_packets:
        answered_paths = packet.get("answered_paths", [])
        if answered_paths:
            lines = [f"实体: {packet['entity']}", "已完成路径:"]
            for idx, path in enumerate(answered_paths, start=1):
                lines.append(
                    f"{idx}. (score={path.get('path_score', 0.0)}) "
                    + render_path_chain(path.get("triplets", []))
                )
            packet_lines.append("\n".join(lines))
            continue

        evidence_lines = packet.get("evidence_lines", [])
        lines = [f"实体: {packet['entity']}"]
        if evidence_lines:
            lines.append("证据:")
            lines.extend(evidence_lines)
        else:
            lines.append("证据: 无")
        packet_lines.append("\n".join(lines))
    return (
        "请严格按照每个实体已经完成的实体证据包，回答原问题。\n"
        "请忠实引用已有证据，完整回答问题，不要有遗漏。"
        '输出格式必须为：{"answer":"...","evidence":"..."}\n'
        f"原问题：{question}\n"
        "实体证据包：\n"
        + "\n\n".join(packet_lines)
    )
