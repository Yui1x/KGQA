#!/usr/bin/env python
"""
E6: build (question, candidate-evidence pool) pairs for validating the joint
sufficiency judgement.

Reviewer 1 asks for "validation on independently annotated complete and
incomplete paths, with precision and recall for false-sufficient decisions".
This script only builds the pairs; it is offline (graph zip + reachability
audit + one logged SCEPS run) and needs neither the API nor Neo4j.
tools/e6_run_judge.py then calls the system's own judgement function once per
pair, and every pair is annotated blind and independently of both the judge
verdict and the construction type.

Pair types
  natural_accepted     pool the full system accepted: its supporting paths plus the
                       other paths it had explored for that chemical (<= 20)
  natural_rejected     pool the full system rejected after an LLM judgement (<= 20)
  complete             gold evidence path(s) plus distractor facts           (expected: sufficient)
  stop_at_bridge       chemical -> intermediate, plus the intermediate's *other*
                       fields; the asked field of the intermediate is absent  (expected: insufficient)
  ownership            chemical -> intermediate is present, but the asked field is
                       offered only as the chemical's own field               (expected: insufficient)
  wrong_field          the asked relation is removed; an adjacent relation of the
                       same chemical is offered                               (expected: insufficient)
  entity_substitution  the asked relation, but from a different chemical whose name
                       contains / is contained in the asked one (the truncation
                       failure mode, e.g. 过氧化氢 -> 氢)                      (expected: insufficient)
  constraint_drop      the question's condition (solution / mother liquor / cylinder /
                       bracket condition such as [粉末]) is not in the evidence
                                                                              (expected: insufficient)

"expected" records the construction intent only. Reported precision / recall
must use the independent annotation, with the construction type as a breakdown.

Usage:
    python tools/e6_build_pairs.py --reach <reachability_v3.csv> --run-dir <SCEPS run raw dir> \
        --out output/e6/pairs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS = ("single_hop", "multi_hop", "hallucination")
TRIP = re.compile(r"^\s*(.*?)\s+-\[(.*)\]->\s+(.*?)\s*$")
COND = re.compile(r"^\[(.+?)\](.+)$")
POOL_CAP = 20
# The asked field of an intermediate -> the chemical's own fields a careless
# judge could accept instead (ownership errors). Same-name relations are tried first.
OWN_FIELD_MAP = {
    "预防措施": ["操作注意事项", "储存注意事项"],
    "防护要求": ["身体防护", "呼吸系统防护", "手防护", "眼睛防护"],
    "危险说明": ["健康危害", "危险特性", "燃爆危险"],
    "贮存要求": ["储存注意事项"],
    "应急防护要求": ["应急处理", "身体防护", "呼吸系统防护"],
    "分类依据": ["危险特性"],
}


def norm(t: str) -> str:
    c = "".join(str(t or "").split())
    return (c.replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":")
             .replace("；", ";").rstrip("。.、,;"))


def fuzzy(g: str, c: str) -> bool:
    g, c = norm(g), norm(c)
    if not g or not c:
        return False
    if g == c:
        return True
    s, l = (g, c) if len(g) <= len(c) else (c, g)
    return (s in l and len(s) >= 3 and len(s) / max(len(l), 1) >= 0.6) or (len(g) >= 2 and g in c)


def gold_parts(gold: str) -> list[str]:
    return [p.strip() for p in re.split(r"[；;/]", str(gold or "")) if p.strip()]


def parse_trip(s: str):
    m = TRIP.match(str(s))
    return m.groups() if m else None


def chain_to_triplets(chain: str) -> list[str]:
    """'A -[r1]-> B -[r2]-> C' -> ['A -[r1]-> B', 'B -[r2]-> C']; other facts kept whole."""
    parts = str(chain).split(" -[")
    if len(parts) < 2:
        return [str(chain).strip()] if str(chain).strip() else []
    head, out = parts[0].strip(), []
    for p in parts[1:]:
        if "]-> " not in p:
            return [str(chain).strip()]
        rel, tail = p.split("]-> ", 1)
        out.append(f"{head} -[{rel}]-> {tail.strip()}")
        head = tail.strip()
    return out


def base_rel(r: str) -> str:
    m = COND.match(r)
    return m.group(2) if m else r


def rel_sim(a: str, b: str) -> float:
    A, B = set(base_rel(a)), set(base_rel(b))
    return len(A & B) / max(1, min(len(A), len(B)))


def load_graph(zip_path: Path):
    edges: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.endswith(".json") or "name_node" in member:
                continue
            for rec in json.loads(zf.read(member).decode("utf-8-sig")):
                h, r, t = (str(rec.get(k) or "").strip() for k in ("head", "relation", "tail"))
                if h and r and t and t not in edges[h][r]:
                    edges[h][r].append(t)
    return edges


def gold_paths(evidence: str):
    m = re.match(r"\s*(\d+)/(\d+) parts \|\s*(.*)$", evidence or "")
    if not m:
        return 0, 0, []
    found, total, rest = int(m.group(1)), int(m.group(2)), m.group(3).split("||")[0]
    trips = [s.strip() for s in rest.split(" | ") if parse_trip(s.strip())]
    paths, i = [], 0
    while i < len(trips):
        h, r, t = parse_trip(trips[i])
        if i + 1 < len(trips) and parse_trip(trips[i + 1])[0] == t:
            paths.append([trips[i], trips[i + 1]])
            i += 2
            continue
        paths.append([trips[i]])
        i += 1
    uniq = []
    for p in paths:
        if p not in uniq:
            uniq.append(p)
    return found, total, uniq


class Builder:
    def __init__(self, edges, rng: random.Random):
        self.edges, self.rng = edges, rng

    def facts(self, c, exclude_rels=(), exclude_tails=(), per_rel=2):
        out = []
        for rel, tails in self.edges.get(c, {}).items():
            if rel in exclude_rels or base_rel(rel) in exclude_rels:
                continue
            kept = [t for t in tails if not any(fuzzy(g, t) for g in exclude_tails)][:per_rel]
            out += [[f"{c} -[{rel}]-> {t}"] for t in kept]
        return out

    def sample(self, items, k):
        items = list(items)
        return self.rng.sample(items, k) if len(items) > k else items

    def mid_other(self, chem, bridge_rel, mid, target_rel, gp, k):
        """2-hop distractors through the gold intermediate, using its other fields."""
        rows = []
        for rel, tails in self.edges.get(mid, {}).items():
            if rel == target_rel or rel_sim(rel, target_rel) >= 0.75:
                continue
            for t in tails[:1]:
                if not any(fuzzy(g, t) for g in gp):
                    rows.append([f"{chem} -[{bridge_rel}]-> {mid}", f"{mid} -[{rel}]-> {t}"])
        return self.sample(rows, k)

    def pool(self, paths):
        seen, out = set(), []
        for p in paths:
            key = " | ".join(p)
            if p and key not in seen:
                seen.add(key)
                out.append({"triplets": p, "path_score": 0.5})
        self.rng.shuffle(out)
        return out[:POOL_CAP]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build E6 sufficiency-validation pairs (offline).")
    ap.add_argument("--data-zip", default=str(PROJECT_DIR / "Data" / "data_after_cluster.zip"))
    ap.add_argument("--reach", required=True, help="reachability audit CSV with evidence_path")
    ap.add_argument("--run-dir", required=True, help="raw dir of one full SCEPS run (single_hop.jsonl ...)")
    ap.add_argument("--out", default=str(PROJECT_DIR / "output" / "e6" / "pairs.jsonl"))
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--per-type", type=int, default=30, help="constructed pairs per type and split")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    edges = load_graph(Path(args.data_zip))
    b = Builder(edges, rng)
    reach = {(r["split"], int(r["qid"])): r for r in csv.DictReader(open(args.reach, encoding="utf-8-sig"))}
    runs = {}
    for split in SPLITS:
        with open(Path(args.run_dir) / f"{split}.jsonl", encoding="utf-8") as fh:
            runs.update({(split, int(json.loads(l)["qid"])): json.loads(l) for l in fh if l.strip()})

    def slots_of(rec):
        s = dict(rec.get("slots") or {})
        return {k: s.get(k) for k in ("chemicals", "event", "bridge", "target", "constraint")}

    def min_hops(slots):
        return 2 if str(slots.get("bridge") or "").strip() else 1

    pairs: list[dict] = []

    def add(kind, split, qid, rec, entity, pool, expected, note, gold=""):
        if not pool:
            return False
        sl = slots_of(rec)
        pairs.append({"type": kind, "split": split, "qid": qid, "question": rec["question"],
                      "gold": gold if split != "hallucination" else "", "entity": entity,
                      "slots": sl, "min_hops": min_hops(sl), "pool": pool,
                      "expected": expected, "note": note})
        return True

    # ---------------- natural pools from the logged run
    nat = {"natural_accepted": defaultdict(list), "natural_rejected": defaultdict(list)}
    for (split, qid), rec in runs.items():
        for pk in rec.get("entity_packets") or []:
            if pk.get("sufficiency_mode") != "joint":
                continue
            explored = [chain_to_triplets(c) for c in pk.get("explored_paths") or []]
            aps = pk.get("answered_paths") or []
            if aps:
                support = [ap["triplets"] for ap in aps if not ap.get("completed_sibling") and ap.get("triplets")]
                nat["natural_accepted"][split].append((qid, rec, pk["entity"], support + explored))
            elif any(j.get("llm_called") and not j.get("sufficient") for j in pk.get("judge_log") or []):
                nat["natural_rejected"][split].append((qid, rec, pk["entity"], explored))
    want = {"natural_accepted": {"single_hop": 30, "multi_hop": 40, "hallucination": 15},
            "natural_rejected": {"single_hop": 15, "multi_hop": 30, "hallucination": 20}}
    for kind, by_split in nat.items():
        for split, items in by_split.items():
            for qid, rec, ent, paths in b.sample(items, want[kind][split]):
                support_note = "system accepted this pool" if kind == "natural_accepted" else "system rejected this pool"
                add(kind, split, qid, rec, ent, b.pool(paths), None, support_note,
                    reach.get((split, qid), {}).get("gold", ""))

    # ---------------- constructed pairs from gold paths
    single, multi = [], []
    for (split, qid), rec in runs.items():
        rr = reach.get((split, qid))
        if not rr or split == "hallucination":
            continue
        found, total, paths = gold_paths(rr.get("evidence_path", ""))
        ent = str(rr.get("resolved_entity") or "")
        if not paths or ent not in edges:
            continue
        gp = gold_parts(rr.get("gold"))
        if split == "single_hop":
            one = [p for p in paths if len(p) == 1 and parse_trip(p[0])[0] == ent]
            if one:
                single.append((qid, rec, ent, one, gp, found == total))
        else:
            two = [p for p in paths if len(p) == 2 and parse_trip(p[0])[0] == ent]
            if two:
                multi.append((qid, rec, ent, two, gp, found == total))
    rng.shuffle(single)
    rng.shuffle(multi)
    n = args.per_type

    def take(items, k, fn):
        made = 0
        for it in items:
            if made >= k:
                break
            made += bool(fn(*it))
        return made

    # complete (positive controls)
    def complete_single(qid, rec, ent, one, gp, full):
        if not full:
            return False
        rel = parse_trip(one[0][0])[1]
        return add("complete", "single_hop", qid, rec, ent,
                   b.pool(one + b.sample(b.facts(ent, exclude_rels={rel, base_rel(rel)}, exclude_tails=gp), 6)),
                   "sufficient", "gold path(s) + 6 distractor facts", rec.get("gold", ""))

    def complete_multi(qid, rec, ent, two, gp, full):
        if not full:
            return False
        _, brel, mid = parse_trip(two[0][0])
        trel = parse_trip(two[0][1])[1]
        extra = [[two[0][0]]] + b.mid_other(ent, brel, mid, trel, gp, 2)
        extra += b.sample(b.facts(ent, exclude_rels={brel, trel}, exclude_tails=gp), 4)
        return add("complete", "multi_hop", qid, rec, ent, b.pool(two + extra), "sufficient",
                   "gold 2-hop path(s) + bridge hop + 2 intermediate-field and 4 chemical distractors",
                   rec.get("gold", ""))

    def stop_at_bridge(qid, rec, ent, two, gp, full):
        _, brel, mid = parse_trip(two[0][0])
        trel = parse_trip(two[0][1])[1]
        others = b.mid_other(ent, brel, mid, trel, gp, 3)
        if not others:
            return False
        paths = [[two[0][0]]] + others + b.sample(b.facts(ent, exclude_rels={brel, trel}, exclude_tails=gp), 4)
        return add("stop_at_bridge", "multi_hop", qid, rec, ent, b.pool(paths), "insufficient",
                   f"reaches {mid} via {brel}; its {trel} is absent (other fields of the intermediate offered)",
                   rec.get("gold", ""))

    def ownership(qid, rec, ent, two, gp, full):
        _, brel, mid = parse_trip(two[0][0])
        trel = base_rel(parse_trip(two[0][1])[1])
        own_rels = [r for r in edges[ent] if base_rel(r) == trel] or \
                   [r for r in edges[ent] if base_rel(r) in OWN_FIELD_MAP.get(trel, [])]
        if not own_rels:
            return False
        own = [[f"{ent} -[{r}]-> {t}"] for r in own_rels[:2] for t in edges[ent][r][:2]]
        paths = [[two[0][0]]] + own + b.mid_other(ent, brel, mid, trel, gp, 2)
        paths += b.sample(b.facts(ent, exclude_rels=set(own_rels) | {brel}, exclude_tails=gp), 3)
        return add("ownership", "multi_hop", qid, rec, ent, b.pool(paths), "insufficient",
                   f"asked field {trel} of {mid} replaced by the chemical's own {', '.join(own_rels[:2])}",
                   rec.get("gold", ""))

    def wrong_field(qid, rec, ent, one, gp, full):
        rel = parse_trip(one[0][0])[1]
        adj = sorted((r for r in edges[ent] if r != rel and base_rel(r) != base_rel(rel) and rel_sim(r, rel) >= 0.5),
                     key=lambda r: -rel_sim(r, rel))
        rows = [[f"{ent} -[{r}]-> {t}"] for r in adj[:2] for t in edges[ent][r][:2]
                if not any(fuzzy(g, t) for g in gp)]
        if not rows:
            return False
        paths = rows + b.sample(b.facts(ent, exclude_rels={rel, base_rel(rel)} | set(adj[:2]), exclude_tails=gp), 5)
        return add("wrong_field", "single_hop", qid, rec, ent, b.pool(paths), "insufficient",
                   f"asked relation {rel} removed; adjacent {', '.join(adj[:2])} offered", rec.get("gold", ""))

    names = sorted(edges, key=len)

    def entity_substitution(qid, rec, ent, one, gp, full):
        rel = parse_trip(one[0][0])[1]
        cands = [c for c in names if c != ent and min(len(c), len(ent)) >= 2 and (c in ent or ent in c)
                 and rel in edges[c] and not any(fuzzy(g, t) for g in gp for t in edges[c][rel])]
        if not cands:
            return False
        sub = max(cands, key=lambda c: (min(len(c), len(ent)), -abs(len(c) - len(ent))))
        rows = [[f"{sub} -[{rel}]-> {t}"] for t in edges[sub][rel][:4]]
        paths = rows + b.sample(b.facts(sub, exclude_rels={rel}), 5)
        return add("entity_substitution", "single_hop", qid, rec, sub, b.pool(paths), "insufficient",
                   f"evidence about {sub} instead of {ent}", rec.get("gold", ""))

    def constraint_bracket(qid, rec, ent, paths_gold, gp, full):
        first = paths_gold[0]
        h, rel, tail = parse_trip(first[0])
        m = COND.match(rel)
        if not m:
            return False
        alts = [r for r in edges[ent] if r != rel and base_rel(r) == m.group(2)]
        if not alts:
            return False
        alt = alts[0]
        rows = []
        for t in edges[ent][alt][:2]:
            if len(first) == 2:
                trel = parse_trip(first[1])[1]
                rows.append([f"{ent} -[{alt}]-> {t}"])
                # 2-hop paths through the wrong-condition intermediate, so that the
                # decision cannot be made by the hop-count precheck alone: the asked
                # field if that intermediate has it, else its closest fields.
                fields = edges.get(t, {})
                rels = [trel] if trel in fields else sorted(fields, key=lambda r: -rel_sim(r, trel))[:2]
                for r2 in rels:
                    for t2 in fields.get(r2, [])[:1]:
                        rows.append([f"{ent} -[{alt}]-> {t}", f"{t} -[{r2}]-> {t2}"])
            else:
                rows.append([f"{ent} -[{alt}]-> {t}"])
        if not rows:
            return False
        paths = rows + b.sample(b.facts(ent, exclude_rels={rel, alt, m.group(2)}, exclude_tails=gp), 4)
        split = "multi_hop" if len(first) == 2 else "single_hop"
        return add("constraint_drop", split, qid, rec, ent, b.pool(paths), "insufficient",
                   f"asked condition {rel} replaced by {alt}", rec.get("gold", ""))

    counts = Counter()
    counts["complete/single"] = take(single, n, complete_single)
    counts["complete/multi"] = take(multi, n, complete_multi)
    counts["stop_at_bridge"] = take(multi, n, stop_at_bridge)
    counts["ownership"] = take(multi, n, ownership)
    counts["wrong_field"] = take(single, n, wrong_field)
    counts["entity_substitution"] = take(single, n, entity_substitution)
    counts["constraint_drop/bracket"] = take(multi + single, n // 3, constraint_bracket)

    # constraint_drop from unanswerable formulation questions: the base chemical's
    # matching field is offered, the condition (solution, mother liquor, cylinder ...) is not
    form = []
    for (split, qid), rec in runs.items():
        if split != "hallucination":
            continue
        log = (rec.get("resolution") or {}).get("log") or []
        bases = [e.get("resolved") for e in log if e.get("rule") == "base_plus_modifier" and e.get("resolved") in edges]
        if len(bases) == 1:
            form.append((qid, rec, bases[0]))
    rng.shuffle(form)
    made = 0
    for qid, rec, base in form:
        if made >= n - counts["constraint_drop/bracket"]:
            break
        target = str((rec.get("slots") or {}).get("target") or "")
        rels = sorted((r for r in edges[base] if target and rel_sim(r, target) >= 0.5), key=lambda r: -rel_sim(r, target))
        if not rels:
            continue
        rows = [[f"{base} -[{r}]-> {t}"] for r in rels[:2] for t in edges[base][r][:2]]
        paths = rows + b.sample(b.facts(base, exclude_rels=set(rels[:2])), 5)
        made += add("constraint_drop", "hallucination", qid, rec, base, b.pool(paths), "insufficient",
                    f"question's condition not in the graph; {base}'s own {', '.join(rels[:2])} offered")
    counts["constraint_drop/formulation"] = made

    for i, p in enumerate(pairs, 1):
        p["pid"] = f"E{i:03d}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    by = Counter((p["type"], p["split"]) for p in pairs)
    for k in sorted(by):
        print(f"  {k[0]:20s} {k[1]:13s} {by[k]:4d}")
    print(f"constructed counts: {dict(counts)}")
    print(f"wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
