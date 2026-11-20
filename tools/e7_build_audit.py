#!/usr/bin/env python
"""
E7: build the knowledge-graph / normalization audit sample (R1: auditable normalization with
merge and non-merge cases linked to source records; R2: audit a sample of triples and clusters).

Every triple in the graph carries the source sentence (`evidence`) and the source document
(`reference`, an MSDS URL or a GB standard), so each audited item is linked to its record.

Three blind sheets (same items for both reviewers, different order):
  A merge audit      original tail -> normalized tail, with a source record    (25 per source file)
  B non-merge audit  two similar tails that normalization kept apart            (~17 per source file)
  C triple audit     head / relation / tail against its source sentence         (25 per source file)

Usage:
    python tools/e7_build_audit.py --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SIX = ["disposal_transport_regulation.json", "fire_and_isolation.json", "first_aid.json",
       "hazard.json", "protection.json", "storage_reactivity.json"]
FIELD_CN = {"disposal_transport_regulation.json": "废弃/运输/法规", "fire_and_isolation.json": "消防与隔离",
            "first_aid.json": "急救", "hazard.json": "危害特性", "protection.json": "防护",
            "storage_reactivity.json": "储存与反应性"}


def load(z, name):
    nm = next(n for n in z.namelist() if n.replace(" ", "") == name)
    return json.loads(z.read(nm).decode("utf-8-sig"))


def norm(s: str) -> str:
    return re.sub(r"[\s，,。.、；;:：（）()\[\]【】]", "", str(s or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=str(PROJECT_DIR / "Data" / "data.zip"))
    ap.add_argument("--after", default=str(PROJECT_DIR / "Data" / "data_after_cluster.zip"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    zb, za = zipfile.ZipFile(args.before), zipfile.ZipFile(args.after)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merges, nonmerges, triples = [], [], []
    for f in SIX:
        B, A = load(zb, f), load(za, f)
        idx = defaultdict(list)
        for r in A:
            idx[(r["head"], r["relation"], r.get("evidence"), r.get("reference"))].append(r["tail"])
        pairs = defaultdict(list)          # (old, new) -> example source records
        for r in B:
            c = idx.get((r["head"], r["relation"], r.get("evidence"), r.get("reference")))
            if c and r["tail"] not in c and len(set(c)) == 1:
                pairs[(r["tail"], c[0])].append(r)
        keys = sorted(pairs); rng.shuffle(keys)
        for old, new in keys[:25]:
            ex = pairs[(old, new)][0]
            merges.append({"field": FIELD_CN[f], "file": f, "head": ex["head"], "relation": ex["relation"], "old": old,
                           "new": new, "evidence": ex.get("evidence", ""), "reference": ex.get("reference", "")})
        # non-merge: similar final strings of the same relation kept apart (blocked on 3-char prefix)
        by_rel = defaultdict(set); example = {}
        for r in A:
            by_rel[r["relation"]].add(r["tail"]); example.setdefault((r["relation"], r["tail"]), r)
        cands = []
        for rel, tails in by_rel.items():
            block = defaultdict(list)
            for t in tails:
                if len(norm(t)) >= 3: block[norm(t)[:3]].append(t)
            for ts in block.values():
                ts = sorted(ts)[:60]
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        a, b = norm(ts[i]), norm(ts[j])
                        if a != b and 0.7 <= min(len(a), len(b)) / max(len(a), len(b)):
                            ratio = difflib.SequenceMatcher(None, a, b).ratio()
                            if ratio >= 0.8:
                                cands.append((rel, ts[i], ts[j], ratio))
        rng.shuffle(cands)
        for rel, t1, t2, ratio in cands[:17]:
            e1, e2 = example[(rel, t1)], example[(rel, t2)]
            nonmerges.append({"field": FIELD_CN[f], "file": f, "relation": rel, "a": t1, "b": t2, "similarity": round(ratio, 2),
                              "head_a": e1["head"], "evidence_a": e1.get("evidence", ""), "head_b": e2["head"], "evidence_b": e2.get("evidence", "")})
        withev = [r for r in A if r.get("evidence")]
        for r in rng.sample(withev, 25):
            triples.append({"field": FIELD_CN[f], "file": f, "head": r["head"], "relation": r["relation"], "tail": r["tail"],
                            "evidence": r.get("evidence", ""), "reference": r.get("reference", "")})
    for kind, rows, prefix in (("merge", merges, "A"), ("nonmerge", nonmerges, "B"), ("triple", triples, "C")):
        for i, r in enumerate(rows, 1):
            r["item_id"] = f"{prefix}{i:03d}"
        with open(out / f"key_{kind}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for who, seed in (("annotator_1", 11), ("annotator_2", 22)):
        r2 = random.Random(seed)
        m = list(merges); r2.shuffle(m)
        with open(out / f"{who}_A_merge.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["编号", "字段类别", "化学品", "关系", "原始说法", "归一化后", "原文句子", "来源", "判定", "改变了什么", "备注"])
            for r in m:
                w.writerow([r["item_id"], r["field"], r["head"], r["relation"], r["old"], r["new"], r["evidence"], r["reference"], "", "", ""])
        n = list(nonmerges); r2.shuffle(n)
        with open(out / f"{who}_B_nonmerge.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["编号", "字段类别", "关系", "说法1", "说法2", "说法1的原文句子", "说法2的原文句子", "判定", "备注"])
            for r in n:
                w.writerow([r["item_id"], r["field"], r["relation"], r["a"], r["b"], f"{r['head_a']}：{r['evidence_a']}", f"{r['head_b']}：{r['evidence_b']}", "", ""])
        t = list(triples); r2.shuffle(t)
        with open(out / f"{who}_C_triple.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["编号", "字段类别", "化学品", "关系", "尾实体", "原文句子", "来源", "判定", "错在哪", "备注"])
            for r in t:
                w.writerow([r["item_id"], r["field"], r["head"], r["relation"], r["tail"], r["evidence"], r["reference"], "", "", ""])
    print(f"merge items: {len(merges)}, non-merge items: {len(nonmerges)}, triple items: {len(triples)} -> {out}")


if __name__ == "__main__":
    main()
