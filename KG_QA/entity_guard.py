"""Conservative chemical-name resolution (entity guard).

Why this module exists
----------------------
The original resolver first accepted any knowledge-graph name that occurred
as a substring of the question, before looking at the parsed mention. Error
analysis of the 2026-11-17 runs showed what that does: 过氧化氢 -> 氢,
次氯酸钠溶液 -> 氯酸钠, 溴化乙锭 -> 溴, 碳酸钠 -> 钠. That is unsafe entity
substitution, and on the unanswerable split it produced every one of the
system's substantive (hallucinated) answers.

Rule
----
1. A parsed mention that equals a graph name or alias after normalisation
   resolves to that chemical.
2. Otherwise the longest graph name contained in the mention is tried. It is
   accepted only if the leftover characters are a recognised form /
   formulation / concentration / item modifier (溶液, 母液, 固体, 无涂层的,
   含量≥60%, 钢瓶, ...). The modifier is returned as a *constraint*, so the
   sufficiency judge has to find a graph fact that grounds it. The graph
   encodes such conditions in relation names, e.g. ``[无涂层]危险性类别`` or
   ``[含量≥60%]危险性类别``.
3. The symmetric case (mention contained in a longer graph name) is accepted
   under the same modifier rule.
4. Anything else is reported as unresolved. The caller treats an unresolved
   mention as missing evidence and abstains for it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

NAME_FIELDS = ("chinese_name", "chinese_alias", "english_name", "english_alias")

# Tokens that may legitimately separate a mention from a graph name. The pattern
# is applied repeatedly until nothing but modifiers is left.
#
# Vocabulary source (revision 2026-11-18): the graph itself encodes conditions
# as bracket prefixes of relation names, and those 13 conditions define what a
# "condition word" is here:
#   乙酸溶液[10%<含量≤25%]  乙酸溶液[25%<含量≤80%]  含量≥60%  20%≤含量<60%
#   8%≤含量<20%  无水或含结晶水<30%  含结晶水≥30%  无涂层  有涂层  干的  湿的
#   粉末  丸状、旋屑或带状
# plus comparators written out in words (大于等于, 不超过, ...), physical state
# words, and generic formulation / item words. The first version omitted the
# written comparators and the dry/wet/state words, so mentions such as
# 含量大于等于60%的双氧水 or 干燥状态下的铪 were left unresolved.
_MODIFIER_PATTERN = re.compile(
    r"含量[为是在]?|浓度[为是在]?|质量分数|含结晶水|结晶水|含水|含|无水|"
    r"大于等于|小于等于|大于|小于|不低于|不高于|不少于|不多于|不超过|超过|低于|高于|"
    r"[≥≤<>＜＞~～\-—–]|\d+(?:\.\d+)?\s*%|"
    r"以上|以下|左右|之间|范围内|在|"
    r"水溶液|稀溶液|浓溶液|溶液|母液|消毒液|"
    r"固体|固态|晶体|结晶|粉末|粉|颗粒|片状|块状|丸状|旋屑|带状|"
    r"气体|液体|液态|液化|气态|液|"
    r"干燥|潮湿|湿润|干|湿|"
    r"[一二三四五六七八九十]?水合物|"
    r"有涂层|无涂层|涂层|"
    r"工业级|试剂级|电子级|分析纯|工业品|商品|产品|"
    r"钢瓶|气瓶|储罐|罐装|罐|桶装|桶|"
    r"状态下|状态|状|"
    r"的|、|或|和|及|与|"
    r"[()\[\]（）【】,，:：\s]"
)
_EDGE_STRIP = "的、,，:：()（）[]【】 \t"


def normalize_name(text: Any) -> str:
    cleaned = "".join(str(text or "").split())
    return (cleaned.replace("（", "(").replace("）", ")")
                   .replace("【", "[").replace("】", "]")
                   .replace("，", ",").replace("：", ":"))


def residual_is_modifier(residual: str) -> bool:
    """True when ``residual`` consists only of whitelisted modifier tokens."""
    if not residual:
        return True
    return _MODIFIER_PATTERN.sub("", residual).strip() == ""


def clean_constraint(residual: str) -> str:
    return residual.strip(_EDGE_STRIP)


def _split_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").replace("；", ";")
    return [part.strip() for part in text.split(";") if part.strip()]


@dataclass
class NameIndex:
    exact: dict[str, str]
    by_length: list[tuple[str, str]]  # (normalized variant, canonical), longest first


@dataclass
class GuardResult:
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)


# Cache keyed by the identity of the (immutable, long-lived) tuple returned by
# HazardNeo4jClient.list_name_records(). The tuple itself is stored alongside
# the index so its id cannot be recycled while the entry exists.
_INDEX_CACHE: dict[int, tuple[tuple, NameIndex]] = {}


def _usable_variant(variant: str) -> bool:
    """Skip very short Latin variants that would match inside unrelated text."""
    if not variant:
        return False
    if variant.isascii():
        return len(variant) >= 3
    return True


def build_name_index(name_records: Iterable[dict[str, Any]]) -> NameIndex:
    if isinstance(name_records, tuple):
        cached = _INDEX_CACHE.get(id(name_records))
        if cached is not None and cached[0] is name_records:
            return cached[1]
    records = tuple(name_records)
    exact: dict[str, str] = {}
    for record in records:
        canonical = str(record.get("chinese_name", "") or "").strip()
        if not canonical:
            continue
        for field_name in NAME_FIELDS:
            for variant in _split_field(record.get(field_name, "")):
                norm = normalize_name(variant)
                if norm and norm not in exact:
                    exact[norm] = canonical
    by_length = sorted(
        ((v, c) for v, c in exact.items() if _usable_variant(v)),
        key=lambda kv: (-len(kv[0]), kv[0]),
    )
    index = NameIndex(exact=exact, by_length=by_length)
    if isinstance(name_records, tuple):
        _INDEX_CACHE[id(name_records)] = (name_records, index)
    return index


def resolve_mention(mention: str, index: NameIndex) -> dict[str, Any]:
    """Resolve one parsed mention. Returns a log entry with a ``rule`` field."""
    norm = normalize_name(mention)
    entry: dict[str, Any] = {"mention": mention, "resolved": None, "constraint": "", "rule": "unresolved"}
    if not norm:
        entry["rule"] = "empty"
        return entry
    hit = index.exact.get(norm)
    if hit:
        entry.update(resolved=hit, rule="exact")
        return entry
    # Longest graph name inside the mention, accepted only with a modifier residual.
    rejected: list[str] = []
    for variant, canonical in index.by_length:
        if len(variant) > len(norm):
            continue
        if variant in norm:
            pos = norm.find(variant)
            prefix, suffix = norm[:pos], norm[pos + len(variant):]
            residual = prefix + suffix
            if residual_is_modifier(residual):
                parts = [p for p in (clean_constraint(prefix), clean_constraint(suffix)) if p]
                entry.update(resolved=canonical, constraint="、".join(parts),
                             rule="base_plus_modifier", matched=variant)
                return entry
            rejected.append(f"{variant}({residual})")
            if len(rejected) >= 3:
                break
    # Mention contained in a longer graph name (e.g. a bracketed formulation).
    for variant, canonical in index.by_length:
        if len(variant) <= len(norm) or len(norm) < 2:
            continue
        if norm in variant and residual_is_modifier(variant.replace(norm, "", 1)):
            entry.update(resolved=canonical, rule="contained_in_graph_name", matched=variant)
            return entry
    if rejected:
        entry["rejected_truncations"] = rejected
    return entry


def guarded_resolve(mentions: Iterable[str], index: NameIndex) -> GuardResult:
    result = GuardResult()
    for mention in mentions:
        entry = resolve_mention(mention, index)
        result.log.append(entry)
        if entry["rule"] == "empty":
            continue
        resolved = entry.get("resolved")
        if not resolved:
            result.unresolved.append(mention)
            continue
        if resolved not in result.entities:
            result.entities.append(resolved)
        constraint = entry.get("constraint") or ""
        if constraint and not any(normalize_name(constraint) == normalize_name(c) for c in result.constraints):
            result.constraints.append(constraint)
    return result


def merge_constraints(existing: list[str], extra: list[str]) -> list[str]:
    """Union of constraint phrases, dropping ones already contained in another."""
    merged: list[str] = []
    for item in list(existing) + list(extra):
        text = str(item or "").strip()
        if not text:
            continue
        norm = normalize_name(text)
        if any(norm == normalize_name(m) or norm in normalize_name(m) for m in merged):
            continue
        merged = [m for m in merged if normalize_name(m) not in norm]
        merged.append(text)
    return merged
