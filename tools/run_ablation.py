#!/usr/bin/env python
"""
Run the SCEPS ablation matrix.

Six configurations, same graph, same test set, same search budget, same
backbone. Each config gets its own resumable run directory:

    output/<tag>/ablation/<variant>/<split>.jsonl

The point of a separate script rather than reusing run_backends.py is that
ablations are single-backbone by construction. Mixing a backbone sweep and an
ablation sweep in one tag would make the output directory ambiguous, and the
paper needs to state "all ablations use one backbone" unambiguously.

A sanity gate runs first: two questions per variant, then it checks that the
variants actually diverge. If every variant produces an identical answer on the
probe, the switches are not wired and the full matrix would waste time and
quota -- so the script stops instead.

Usage:
    python tools/run_ablation.py --tag abl-ds --model deepseek-flash \\
        --api-base https://api.deepseek.com/v1 --api-key ... --concurrency 200

    python tools/run_ablation.py --tag abl-ds --model deepseek-flash ... \\
        --variants wo_slot_coverage,wo_abstention    # subset

    python tools/run_ablation.py ... --skip-probe     # once the wiring is trusted
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_DIR / "tools" / "run_parallel.py"

# Variant name -> extra arguments for tools/run_parallel.py. The first six are
# component ablations (--ablation); the last four isolate the revision's
# changes, so every change can be reported as its own row.
VARIANT_FLAGS: dict[str, list[str]] = {
    "full": ["--ablation", "full"],
    "wo_slot_coverage": ["--ablation", "wo_slot_coverage"],
    "wo_sufficiency_check": ["--ablation", "wo_sufficiency_check"],
    "wo_abstention": ["--ablation", "wo_abstention"],
    "entity_target_only": ["--ablation", "entity_target_only"],
    "wo_constraint_slot": ["--ablation", "wo_constraint_slot"],
    "per_path_judge": ["--ablation", "full", "--sufficiency-mode", "per_path"],
    "wo_structural_rule": ["--ablation", "full", "--structural-rule", "off"],
    "wo_entity_guard": ["--ablation", "full", "--entity-guard", "off"],
    "with_repair": ["--ablation", "full", "--repair", "on"],
    "wo_field_completion": ["--ablation", "full", "--field-completion", "off"],
    "wo_slot_guided_expansion": ["--ablation", "full", "--slot-guided-expansion", "off"],
}
VARIANTS = list(VARIANT_FLAGS)


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def probe(tag: str, model: str, base: str, key: str, neo4j_pw: str,
          neo4j_uri: str, neo4j_user: str) -> bool:
    """Two questions per variant; verify the variants produce different output."""
    print("=" * 88)
    print("WIRING PROBE: 2 questions per variant")
    print("=" * 88)
    signatures: dict[str, str] = {}
    for variant in VARIANTS:
        probe_tag = f"{tag}/_probe/{variant}"
        cmd = [
            sys.executable, str(RUNNER),
            "--splits", "multi_hop",
            "--limit", "2",
            "--concurrency", "2",
            "--run-tag", probe_tag,
            "--model", model,
            *VARIANT_FLAGS[variant],
            "--api-base", base,
            "--api-key", key,
            "--neo4j-uri", neo4j_uri,
            "--neo4j-user", neo4j_user,
            "--neo4j-password", neo4j_pw,
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_DIR),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  {variant:22s} PROBE FAILED")
            print(proc.stdout[-600:])
            print(proc.stderr[-600:])
            return False
        path = PROJECT_DIR / "output" / probe_tag / "multi_hop.jsonl"
        rows = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        # Signature = (abstained flags, answer prefixes). Two variants that agree
        # on both are behaviourally identical on the probe.
        sig = "|".join(
            f"{int(bool(r.get('abstained')))}:{str(r.get('answer', ''))[:40]}"
            for r in rows
        )
        signatures[variant] = sig
        states = [bool(r.get("abstained")) for r in rows]
        print(f"  {variant:22s} n={len(rows)} abstained={states} "
              f"calls={[r.get('usage', {}).get('request_count') for r in rows]}")

    distinct = len(set(signatures.values()))
    print(f"\n  distinct behaviour signatures: {distinct}/{len(VARIANTS)}")
    if distinct <= 1:
        print("  ABORT: all variants behave identically -- the switches are not wired")
        return False
    print("  probe passed: variants diverge")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SCEPS ablation matrix.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--splits", default="single_hop,multi_hop,hallucination")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--requests-per-minute", type=int, default=0)
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE",
                                                        "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", ""))
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants: {unknown}")
    if not args.api_key:
        sys.exit("need --api-key or DEEPSEEK_API_KEY")
    if not args.neo4j_password:
        sys.exit("need --neo4j-password or KGQA_NEO4J_PASSWORD")

    print(f"ablation sweep: {args.tag}")
    print(f"  model      : {args.model}")
    print(f"  variants   : {', '.join(variants)}")
    print(f"  splits     : {args.splits}")
    print(f"  concurrency: {args.concurrency}")

    if args.dry_run:
        for variant in variants:
            base_dir = Path(args.output_dir) / args.tag / "ablation" / variant
            done = {s: count_rows(base_dir / f"{s}.jsonl")
                    for s in args.splits.split(",")}
            print(f"    {variant:22s} done={done}")
        print("\n(dry run: nothing executed)")
        return

    if not args.skip_probe:
        ok = probe(args.tag, args.model, args.api_base, args.api_key,
                   args.neo4j_password, args.neo4j_uri, args.neo4j_user)
        if not ok:
            sys.exit(1)

    sweep_started = time.time()
    failures: list[tuple[str, int]] = []
    for index, variant in enumerate(variants, start=1):
        run_tag = f"{args.tag}/ablation/{variant}"
        print(f"\n{'=' * 88}\n[{index}/{len(variants)}] {variant}\n{'=' * 88}")
        started = time.time()
        cmd = [
            sys.executable, str(RUNNER),
            "--splits", args.splits,
            "--output-dir", args.output_dir,
            "--run-tag", run_tag,
            "--concurrency", str(args.concurrency),
            "--requests-per-minute", str(args.requests_per_minute),
            "--model", args.model,
            *VARIANT_FLAGS[variant],
            "--api-base", args.api_base,
            "--api-key", args.api_key,
            "--neo4j-uri", args.neo4j_uri,
            "--neo4j-user", args.neo4j_user,
            "--neo4j-password", args.neo4j_password,
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_DIR))
        elapsed = time.time() - started
        if proc.returncode != 0:
            failures.append((variant, proc.returncode))
            print(f"  !! failed (exit {proc.returncode}) after {elapsed/60:.1f}m; continuing")
        else:
            print(f"  done in {elapsed/60:.1f}m")

    print(f"\n{'=' * 88}\nablation sweep finished in {(time.time()-sweep_started)/60:.1f} min")
    if failures:
        print("FAILED VARIANTS:")
        for variant, code in failures:
            print(f"  {variant} -> exit {code}")
        print("re-run the same command to resume")
    else:
        print("all variants completed")

    print("\nper-variant counts:")
    for variant in variants:
        base_dir = Path(args.output_dir) / args.tag / "ablation" / variant
        done = {s: count_rows(base_dir / f"{s}.jsonl")
                for s in args.splits.split(",")}
        print(f"  {variant:22s} {done}")


if __name__ == "__main__":
    main()
