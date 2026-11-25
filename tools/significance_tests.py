"""Paired significance tests for the main comparison (Table 4).

Reads the released per-question labels and reports, for every baseline and
answerable split, the exact McNemar test against each SCEPS run, the Holm
correction over the 14 baseline-split comparisons, taking for each comparison
the largest p-value of the three runs.

Usage:
    python tools/significance_tests.py [path/to/labels_per_question.csv]
"""
import csv
import sys
from math import comb
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "Result" / "Revision" / "labels" / "labels_per_question.csv"
BACKBONE = "deepseek-v4.1-flash"
RUNS = ["sceps_run1", "sceps_run2", "sceps_run3"]
BASELINES = ["zero_shot", "few_shot", "cot", "kaping", "tog", "tog2", "search_on_graph"]
SPLITS = ["single_hop", "multi_hop"]


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    correct = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["backbone"] == BACKBONE and r["split"] in SPLITS:
                correct[(r["system"], r["split"], r["qid"])] = r["answer_label"] == "correct"

    def test(sys_a, sys_b, split):
        qids = [q for (s, sp, q) in correct if s == sys_a and sp == split]
        b = sum(correct[(sys_a, split, q)] and not correct[(sys_b, split, q)] for q in qids)
        c = sum(correct[(sys_b, split, q)] and not correct[(sys_a, split, q)] for q in qids)
        return b, c, exact_mcnemar(b, c)

    rows = []
    for split in SPLITS:
        for base in BASELINES:
            per_run = [test(run, base, split) for run in RUNS]
            rows.append((split, base, per_run, max(p for _, _, p in per_run)))

    adjusted = holm([r[3] for r in rows])
    print(f"{'split':<11} {'baseline':<16} {'p run1':>10} {'p run2':>10} {'p run3':>10} {'max p':>10} {'Holm (14)':>10}")
    for (split, base, per_run, pmax), padj in zip(rows, adjusted):
        ps = " ".join(f"{p:10.2e}" for _, _, p in per_run)
        print(f"{split:<11} {base:<16} {ps} {pmax:10.2e} {padj:10.2e}")
    for split in SPLITS:
        worst = max(a for (sp, *_), a in zip(rows, adjusted) if sp == split)
        print(f"Largest Holm-adjusted p ({split}, 14 comparisons): {worst:.3g}")


if __name__ == "__main__":
    main()
