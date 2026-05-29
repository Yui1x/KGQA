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
    return (
        "请对下面问题做轻量槽位化解析。\n"
        "要求：\n"
        "1. chemicals: 只抽取问题中直接出现的化学品，数量不固定，可以是0个、1个或多个。\n"
        "2. 必须保留题目中的完整化学品写法，不能丢掉括号、位次词、前后缀等表面形式。\n"
        "   例如：(对)异丙基苯酚 不能简化成 异丙基苯酚。\n"
        "3. event: 问题场景事件，例如 起火、泄漏、皮肤接触、眼睛接触、吸入；没有就写空字符串。\n"
        "4. bridge: 中间环节，例如 燃烧产物、危险性类别、反应产物；没有就写空字符串。\n"
        "5. target: 最终关注目标，例如 症状、严重后果、处置措施、防护要求、监测方式；没有就写空字符串。\n"
        "6. constraint: 题目中的限制词列表，例如 最早、一般、严重、立即；没有就返回空列表。\n"
        "7. 不要添加图谱里可能存在但题目没有出现的实体。\n"
        '8. 只输出JSON，例如 {"chemicals":["1,6-己二胺"],"event":"起火","bridge":"燃烧产物","target":"吸入症状","constraint":["最早","一般"]}。\n'
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
