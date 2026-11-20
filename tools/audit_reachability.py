#!/usr/bin/env python
"""
Gold-answer reachability audit against the live Neo4j graph.

Answers three review questions directly:

  * R2-M5  "how can accuracy exceed path completeness?"  -> by measuring how
           many questions in each split are actually answerable in 1 hop.
           A "multi-hop" question that is 1-hop answerable cannot have a
           complete multi-hop path, yet can be answered correctly.
  * R1     "how are missing facts separated from retrieval failure?" -> the
           `unreachable` stratum is fact-missing; questions whose gold answer
           is present within d_max but were still abstained are retrieval
           failures.
  * R2-M1  "how were unanswerable cases verified against the KG?" -> the same
           audit run over the hallucination split.

Matching policy is conservative and documented here so the numbers can be
reproduced by hand:

  1. exact string equality, trimmed;
  2. containment in either direction, but only when the shorter string has at
     least MIN_SUBSTR chars AND covers at least MIN_RATIO of the longer one.
     This rejects the pathological '水' in '如沾染衣服:立即用水...' match that a
     naive substring test produces.

Usage:
    python tools/audit_reachability.py --out analysis/
    python tools/audit_reachability.py --no-db     # use the data zip instead
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MIN_SUBSTR = 3
MIN_RATIO = 0.60
MAX_DEPTH = 3

# Only the canonical name is excluded from the property search: a question like
# "乙酸丙酯的中文别名有什么？" has its gold answer in `中文别名`, so alias and
# English-name fields must stay searchable or those questions look unreachable.
EXCLUDED_PROPERTY = ("中文名称",)

# Multi-hop gold answers are frequently unions of several graph tails joined by
# a Chinese semicolon, e.g. three GB 30000 precaution phrases concatenated.
# An answer counts as reachable only when EVERY part is locatable, so the split
# separator is part of the metric definition, not a preprocessing detail.
GOLD_SEPARATORS = ("；", ";")

# Two matching policies are reported side by side, because the reachability
# figure moves a lot with the string rule and a reviewer will ask. Definitions
# are fixed here instead of being tuned until the numbers look good.
#
#   strict  -- exact, or containment at >= MIN_RATIO of the longer string.
#              Conservative; understates reachability for values embedded in a
#              longer phrase ("棕色" inside "黄色吸水性结晶，在空气中能转变成棕色。").
#   relaxed -- additionally accepts:
#                * a numeric gold matched against the leading numeric token of a
#                  property, e.g. '53.32' vs '53.32(41.6℃)';
#                * a CJK gold of >= 2 characters appearing anywhere in the value,
#                  e.g. '棕色' or '有类似葡萄酒的气味'.
#              Single-character CJK is always rejected: it is what produced the
#              pathological '水' inside '立即用水充分冲洗' match.
NUMERIC_RE = re.compile(r"^[\d.,~<>≤≥=\-]+")
CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")


# ------------------------------------------------------------------ graph ---

def load_graph_from_neo4j(uri: str, user: str, password: str) -> tuple[dict, dict]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    chems: dict[str, dict[str, Any]] = {}
    edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    try:
        with driver.session() as session:
            for rec in session.run(
                "MATCH (c:Chemical) RETURN properties(c) AS p"
            ):
                props = dict(rec["p"])
                name = str(props.get("中文名称") or "").strip()
                if name:
                    chems[name] = props
            for rec in session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE h:Chemical OR h:hazard_category OR h:gb30000
                RETURN coalesce(h.中文名称, h.name) AS head,
                       type(r) AS rel,
                       coalesce(t.中文名称, t.name) AS tail
                """
            ):
                head = str(rec["head"] or "").strip()
                tail = str(rec["tail"] or "").strip()
                if head and tail:
                    edges[head].append((str(rec["rel"]), tail))
    finally:
        driver.close()
    return chems, edges


def load_graph_from_zip(zip_path: Path) -> tuple[dict, dict]:
    chems: dict[str, dict[str, Any]] = {}
    edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for rec in json.loads(zf.read("name_node_properties.json").decode("utf-8-sig")):
            name = str(rec.get("中文名称") or "").strip()
            if name:
                chems[name] = rec
        for member in zf.namelist():
            if not member.endswith(".json") or "name_node" in member:
                continue
            for rec in json.loads(zf.read(member).decode("utf-8-sig")):
                head = str(rec.get("head") or "").strip()
                tail = str(rec.get("tail") or "").strip()
                rel = str(rec.get("relation") or "").strip()
                if head and tail and rel:
                    edges[head].append((rel, tail))
    return chems, edges


# ---------------------------------------------------------------- matching ---

def matches(gold: str, candidate: str, policy: str = "strict") -> bool:
    gold = normalize(gold).rstrip("。.、,;")
    cand = normalize(candidate).rstrip("。.、,;")
    if not gold or not cand:
        return False
    if gold == cand:
        return True
    short, long = (gold, cand) if len(gold) <= len(cand) else (cand, gold)
    if short in long and len(short) >= MIN_SUBSTR and len(short) / max(len(long), 1) >= MIN_RATIO:
        return True
    if policy == "relaxed":
        if numeric_equivalent(gold, cand):
            return True
        if CJK_RE.match(gold) and len(gold) >= 2 and gold in cand:
            return True
        if long_text_match(gold, cand):
            return True
    return False


_PUNCT_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


def long_text_match(gold: str, cand: str) -> bool:
    """Punctuation- and width-insensitive containment for long answers.

    Revision 2026-11-18: GHS classification criteria and label descriptions are
    long strings whose punctuation differs between the question set and the
    graph (full/half-width commas, spaces, ':'), so the 60%-coverage rule
    reported such answers as unreachable even when the graph held them.
    """
    g, c = _PUNCT_RE.sub("", gold).lower(), _PUNCT_RE.sub("", cand).lower()
    if len(g) < 6 or not c:
        return False
    if g in c:
        return True
    return len(c) >= 6 and c in g and len(c) / len(g) >= 0.6


def numeric_equivalent(gold: str, cand: str) -> bool:
    """Same leading number, and units compatible when both carry one.

    Revision 2026-11-17: the earlier rule required the *whole* gold string to
    be numeric and did not accept a leading "+", so '99.2kJ/kg' vs '+99.2',
    '589l/kg' vs '589 L/kg' and similar values were reported as unreachable.
    """
    g = gold.strip().lstrip("+").lower()
    c = cand.strip().lstrip("+").lower()
    g_lead, c_lead = NUMERIC_RE.match(g), NUMERIC_RE.match(c)
    if not g_lead or not c_lead:
        return False
    g_num, c_num = g_lead.group(0).rstrip(".,"), c_lead.group(0).rstrip(".,")
    if not g_num or g_num != c_num or not any(ch.isdigit() for ch in g_num):
        return False
    g_unit = g[g_lead.end():].strip(" ()（）")
    c_unit = c[c_lead.end():].strip(" ()（）")
    if g_unit and c_unit and g_unit not in c_unit and c_unit not in g_unit:
        return False
    return True


def normalize(text: str) -> str:
    """Collapse whitespace and unify full/half-width punctuation.

    The graph stores '，' and '(' while the question set mixes both widths; without
    normalisation an otherwise identical phrase fails the exact test.
    """
    cleaned = "".join(str(text or "").split())
    return (cleaned.replace("（", "(").replace("）", ")")
                   .replace("，", ",").replace("：", ":")
                   .replace("；", ";"))


def split_gold(gold: str) -> list[str]:
    """Split a composed gold answer into its constituent phrases."""
    parts: list[str] = []
    for chunk in [gold]:
        pieces = [chunk]
        for sep in GOLD_SEPARATORS:
            pieces = [p for piece in pieces for p in piece.split(sep)]
        parts.extend(p for p in (x.strip() for x in pieces) if p)
    return parts or [gold]


def build_name_variants(chems: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(variant, canonical) pairs over Chinese names and aliases, longest first.

    Revision 2026-11-17: the audit used canonical names only, so a question
    phrased with an alias (过氧化氢 for 双氧水, 氢氧化钠 for 烧碱) counted as
    "no chemical resolved", and the shortest-name problem (过氧化氢 -> 氢) was
    possible. Longest-variant-first matching over aliases fixes both.
    """
    pairs: dict[str, str] = {}
    for name, rec in chems.items():
        pairs.setdefault(name, name)
        for alias in re.split(r"[;；]", str(rec.get("中文别名") or "")):
            alias = alias.strip()
            if len(alias) >= 2:
                pairs.setdefault(alias, name)
    return sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def resolve_chemical(question: str, names: list) -> str | None:
    """Canonical name of the longest graph name/alias contained in the question."""
    for item in names:
        variant, canonical = item if isinstance(item, tuple) else (item, item)
        if variant and variant in question:
            return canonical
    return None


def property_values(props: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for key, value in props.items():
        if key in EXCLUDED_PROPERTY or not isinstance(value, str):
            continue
        if value.strip():
            out.append((key, value.strip()))
    return out


def locate(
    phrase: str,
    entity: str,
    chems: dict[str, dict[str, Any]],
    edges: dict[str, list[tuple[str, str]]],
    policy: str = "strict",
) -> dict[str, Any] | None:
    """Find one phrase, preferring the shortest path from `entity`."""
    for name, value in property_values(chems.get(entity, {})):
        if matches(phrase, value, policy):
            return {"hop": 0, "via": "property", "path": [f"{entity}.{name}"]}

    frontier: list[tuple[str, list[str]]] = [(entity, [])]
    visited = {entity}
    for depth in range(1, MAX_DEPTH + 1):
        nxt: list[tuple[str, list[str]]] = []
        for node, path in frontier:
            for rel, tail in edges.get(node, []):
                trail = path + [f"{node} -[{rel}]-> {tail}"]
                if matches(phrase, tail, policy):
                    return {"hop": depth, "via": "edge", "path": trail}
                if tail not in visited:
                    visited.add(tail)
                    nxt.append((tail, trail))
        frontier = nxt
        if not frontier:
            break
    return None


def find_hops(
    question: str,
    gold: str,
    entity: str,
    chems: dict[str, dict[str, Any]],
    edges: dict[str, list[tuple[str, str]]],
    policy: str = "strict",
) -> dict[str, Any]:
    """A gold answer is reachable only when every one of its parts is reachable.

    The reported depth is the deepest part, because a composed answer needs the
    full chain to be reproduced.
    """
    parts = split_gold(gold)
    located = []
    for phrase in parts:
        hit = locate(phrase, entity, chems, edges, policy)
        if hit is None:
            return {"hop": None, "via": None, "path": [], "missing": phrase,
                    "parts": len(parts), "found": len(located)}
        located.append(hit)
    deepest = max(located, key=lambda h: h["hop"])
    return {
        "hop": deepest["hop"],
        "via": deepest["via"],
        "path": [p for h in located for p in h["path"]],
        "missing": "",
        "parts": len(parts),
        "found": len(located),
    }


# ------------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-answer reachability audit.")
    parser.add_argument("--qa-dir", default="Question&Answer")
    parser.add_argument("--data-zip", default="Data/data_after_cluster.zip")
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--no-db", action="store_true", help="use the data zip instead of Neo4j")
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", ""))
    args = parser.parse_args()

    if args.no_db:
        print(f"loading graph from {args.data_zip} ...")
        chems, edges = load_graph_from_zip(Path(args.data_zip))
    else:
        if not args.neo4j_password:
            sys.exit("need --neo4j-password or KGQA_NEO4J_PASSWORD (or pass --no-db)")
        print(f"loading graph from {args.neo4j_uri} ...")
        chems, edges = load_graph_from_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    print(f"  chemicals={len(chems)}  heads_with_edges={len(edges)}  "
          f"edges={sum(len(v) for v in edges.values())}")
    names = build_name_variants(chems)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[list[str]] = []
    strata: dict[str, dict[str, Counter[str]]] = {}

    for split in ("single_hop", "multi_hop", "hallucination"):
        with (Path(args.qa_dir) / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as fh:
            questions = list(csv.DictReader(fh))
        strata[split] = {"strict": Counter(), "relaxed": Counter()}
        for qid, rec in enumerate(questions, start=1):
            question = (rec.get("Questions") or "").strip()
            gold = (rec.get("Answers") or "").strip()
            entity = resolve_chemical(question, names)
            if not entity:
                for policy in ("strict", "relaxed"):
                    strata[split][policy]["no_entity_resolved"] += 1
                rows_out.append([split, str(qid), question, gold, "", "no_entity", "", "", ""])
                continue
            strict = find_hops(question, gold, entity, chems, edges, "strict")
            relaxed = find_hops(question, gold, entity, chems, edges, "relaxed")

            def classify(hit: dict[str, Any], policy: str) -> str:
                hop = hit["hop"]
                if hop is None:
                    counter = strata[split][policy]
                    counter[f"unreachable@{MAX_DEPTH}"] += 1
                    return f"unreachable@{MAX_DEPTH}"
                counter = strata[split][policy]
                if hit["via"] == "property":
                    counter["property"] += 1
                    return "property"
                counter[f"edge_{hop}hop"] += 1
                return f"edge_{hop}hop"

            strict_label = classify(strict, "strict")
            relaxed_label = classify(relaxed, "relaxed")
            hop = strict["hop"]
            rows_out.append([
                split, str(qid), question, gold, entity, strict_label, relaxed_label,
                str(hop if hop is not None else ""),
                f'{strict["found"]}/{strict["parts"]} parts | ' + " | ".join(strict["path"])[:330]
                + (f' || MISSING: {strict["missing"][:80]}' if strict["missing"] else ""),
            ])

    with (out_dir / "reachability.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "qid", "question", "gold", "resolved_entity",
                         "class_strict", "class_relaxed", "hop", "evidence_path"])
        writer.writerows(rows_out)

    buckets = ("property", "edge_1hop", "edge_2hop", "edge_3hop",
               f"unreachable@{MAX_DEPTH}", "no_entity_resolved")
    for policy in ("strict", "relaxed"):
        rule = ("exact, or substr>=%d chars and >=%d%% of longer"
                % (MIN_SUBSTR, MIN_RATIO * 100)) if policy == "strict" else \
               "strict + numeric-token + CJK>=2 substring"
        print(f"\n{'=' * 104}\nGOLD-ANSWER REACHABILITY [{policy}]  rule: {rule}\n{'=' * 104}")
        print(f"{'split':14s} {'n':>4s}  " + "  ".join(f"{k:>18s}" for k in buckets))
        for split, per_policy in strata.items():
            counter = per_policy[policy]
            total = sum(counter.values())
            cells = [f"{counter.get(k, 0):4d} ({100*counter.get(k,0)/total:5.1f}%)" for k in buckets]
            print(f"{split:14s} {total:4d}  " + "  ".join(cells))

    mh = strata["multi_hop"]["relaxed"]
    mh_total = sum(mh.values())
    one_hop = mh.get("property", 0) + mh.get("edge_1hop", 0)
    print(f"\nR2-M5 basis (relaxed rule): {one_hop}/{mh_total} multi-hop questions "
          f"({100*one_hop/mh_total:.1f}%) are answerable in at most one hop.")
    print("  -> those cannot have a complete *multi-hop* path by construction,")
    print("     so accuracy can exceed multi-hop path completeness without any fallback.")

    print(f"\nwrote {out_dir / 'reachability.csv'}")


if __name__ == "__main__":
    main()
