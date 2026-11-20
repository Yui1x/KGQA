#!/usr/bin/env python
"""
Plan and launch the multi-backbone sweep with a split backend configuration.

Two endpoints are involved:

  * the workbuddy gateway (key A) -- serves the non-DeepSeek backbones
  * the DeepSeek API (key B)      -- serves deepseek-v4.1-flash

Both are OpenAI-compatible, so the only per-backend difference is base URL,
key, and model id. This script keeps that mapping in one place and refuses to
start a cell whose credentials are missing, rather than letting it run degraded.

Design notes that matter for the paper:

  * The reasoning knob is per-call-site, not per-run. Parsing, relation ranking
    and candidate filtering send reasoning_effort="none"; the sufficiency judge
    and the final aggregation keep the backend default. utils.resolve_reasoning
    additionally gates "none" to the models that actually honour it, because on
    the glm family it consumes the whole budget and returns empty content.
  * Every cell is resumable. A finished question is never re-requested.
  * Pacing is per cell, defaulting to just under the gateway's 120 req/min.

Usage:
    python tools/run_backends.py --tag paper-v2 --models cn:mimo-v2.6-pro,cn:kimi-k2.5 \\
        --gateway-key ... --deepseek-key ... --concurrency 10 --requests-per-minute 110

    # add deepseek later, same tag; finished cells are skipped
    python tools/run_backends.py --tag paper-v2 --models deepseek-v4.1-flash \\
        --deepseek-key ... --concurrency 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_DIR / "tools" / "run_parallel.py"

# model id -> backend name. Anything not listed defaults to the gateway.
# The official DeepSeek API uses different ids than the gateway ("deepseek-flash"
# rather than "cn:deepseek-v4.1-flash"), so both spellings are mapped.
BACKEND_OF = {
    "deepseek-flash": "deepseek",
    "deepseek-v4.1-flash": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "deepseek-v4-pro": "deepseek",
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
}

DEFAULT_GATEWAY_BASE = "http://127.0.0.1:7864/v1"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com/v1"

# Pacing defaults differ by backend by four orders of magnitude: the local
# gateway sheds load above ~120 req/min, while the official DeepSeek API
# tolerates very high concurrency. Applying the gateway's pacing to DeepSeek
# would turn a 5-minute cell into an hour.
DEFAULT_PACING = {"gateway": 110, "deepseek": 0}
DEFAULT_CONCURRENCY = {"gateway": 10, "deepseek": 64}


def slug(text: str) -> str:
    return text.replace(":", "-").replace(".", "_").replace("/", "_")


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a multi-backend sweep.")
    parser.add_argument("--models", required=True, help="comma-separated model ids")
    parser.add_argument("--splits", default="single_hop,multi_hop,hallucination")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--concurrency", type=int, default=0,
                        help="per-cell workers; 0 = backend default "
                             f"({DEFAULT_CONCURRENCY})")
    parser.add_argument("--requests-per-minute", type=int, default=-1,
                        help="per-cell request ceiling; -1 = backend default "
                             f"({DEFAULT_PACING}), 0 = unlimited")
    parser.add_argument("--gateway-key", default=os.getenv("KGQA_API_KEY", ""))
    parser.add_argument("--gateway-base", default=os.getenv("KGQA_API_BASE", DEFAULT_GATEWAY_BASE))
    parser.add_argument("--deepseek-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--deepseek-base", default=os.getenv("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_BASE))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Resolve each model to its backend up front so a missing key fails loudly
    # before any quota is spent.
    plan: list[tuple[str, str, str, str]] = []
    missing: list[str] = []
    for model in models:
        backend = BACKEND_OF.get(model, "gateway")
        if backend == "deepseek":
            base, key = args.deepseek_base, args.deepseek_key
            if not key:
                missing.append(f"{model} needs --deepseek-key")
                continue
        else:
            base, key = args.gateway_base, args.gateway_key
            if not key:
                missing.append(f"{model} needs --gateway-key")
                continue
        plan.append((model, backend, base, key))

    if missing:
        for line in missing:
            print(f"MISSING CREDENTIAL: {line}", file=sys.stderr)
        sys.exit(1)
    if not args.neo4j_password:
        sys.exit("need --neo4j-password or KGQA_NEO4J_PASSWORD")

    print(f"sweep tag: {args.tag}")
    print(f"  splits      : {', '.join(splits)}")
    print(f"  repeats     : {args.repeats}")
    print("  models:")
    for model, backend, base, _ in plan:
        conc = args.concurrency or DEFAULT_CONCURRENCY[backend]
        pace = DEFAULT_PACING[backend] if args.requests_per_minute < 0 else args.requests_per_minute
        print(f"    {model:26s} -> {backend:9s} {base}")
        print(f"      concurrency={conc}  pacing={pace or 'unlimited'} req/min")

    if args.dry_run:
        for model, backend, base, key in plan:
            for rep in range(1, args.repeats + 1):
                base_dir = Path(args.output_dir) / args.tag / slug(model) / f"r{rep}"
                done = {s: count_rows(base_dir / f"{s}.jsonl") for s in splits}
                print(f"    {slug(model):26s} r{rep} done={done}")
        print("\n(dry run: nothing executed)")
        return

    sweep_started = time.time()
    failures: list[tuple[str, int, int]] = []
    total_cells = len(plan) * args.repeats
    cell_index = 0

    for model, backend, base, key in plan:
        for rep in range(1, args.repeats + 1):
            cell_index += 1
            run_tag = f"{args.tag}/{slug(model)}/r{rep}"
            concurrency = args.concurrency or DEFAULT_CONCURRENCY[backend]
            pacing = DEFAULT_PACING[backend] if args.requests_per_minute < 0 else args.requests_per_minute
            print(f"\n{'=' * 88}\n[{cell_index}/{total_cells}] {model} ({backend}) "
                  f"repeat {rep}\n  concurrency={concurrency} pacing={pacing or 'unlimited'}"
                  f"\n  -> {run_tag}\n{'=' * 88}")
            cell_started = time.time()
            cmd = [
                sys.executable, str(RUNNER),
                "--splits", ",".join(splits),
                "--output-dir", args.output_dir,
                "--run-tag", run_tag,
                "--concurrency", str(concurrency),
                "--requests-per-minute", str(pacing),
                "--model", model,
                "--entity-model", model,
                "--api-base", base,
                "--api-key", key,
                "--neo4j-uri", args.neo4j_uri,
                "--neo4j-user", args.neo4j_user,
                "--neo4j-password", args.neo4j_password,
            ]
            proc = subprocess.run(cmd, cwd=str(PROJECT_DIR))
            elapsed = time.time() - cell_started
            if proc.returncode != 0:
                failures.append((model, rep, proc.returncode))
                print(f"  !! failed (exit {proc.returncode}) after {elapsed/60:.1f}m; continuing")
            else:
                print(f"  cell done in {elapsed/60:.1f}m")

    print(f"\n{'=' * 88}\nsweep finished in {(time.time()-sweep_started)/60:.1f} min")
    if failures:
        print("FAILED CELLS:")
        for model, rep, code in failures:
            print(f"  {model} r{rep} -> exit {code}")
        print("re-run the same command to resume")
    else:
        print("all cells completed")

    print("\nper-cell counts:")
    for model, _backend, _base, _key in plan:
        for rep in range(1, args.repeats + 1):
            base_dir = Path(args.output_dir) / args.tag / slug(model) / f"r{rep}"
            done = {s: count_rows(base_dir / f"{s}.jsonl") for s in splits}
            print(f"  {slug(model):26s} r{rep}  {done}")


if __name__ == "__main__":
    main()
