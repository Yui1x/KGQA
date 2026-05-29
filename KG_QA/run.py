from __future__ import annotations

import os
from pathlib import Path

from main import run_from_config





PROJECT_DIR = Path(__file__).resolve().parent
KGQA_DIR = PROJECT_DIR.parent

API_BASE = os.getenv("KGQA_API_BASE", os.getenv("OPENAI_BASE_URL", ""))
API_KEY = os.getenv("KGQA_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MODEL = os.getenv("KGQA_MODEL", "qwen-plus-2025-12-01")

NEO4J_URI = os.getenv("KGQA_NEO4J_URI", os.getenv("NEO4J_URI", ""))
NEO4J_USER = os.getenv("KGQA_NEO4J_USER", os.getenv("NEO4J_USER", ""))
NEO4J_PASSWORD = os.getenv("KGQA_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", ""))




SINGLE_QUESTION = ""
START_QUESTION = 1
SAMPLES = 99999

INPUT_CSV = KGQA_DIR / "Question&Answer"
INPUT_SPLIT = ["single_hop", "multi_hop"]


OUTPUT_DIR = KGQA_DIR / "output" / f"{MODEL}_output"

DEPTH = 3
WIDTH = 3
MAX_LENGTH = 768
TEMPERATURE_EXPLORATION = 0.0
TEMPERATURE_REASONING = 0.0
CANDIDATE_POOL_SIZE = 8
MAX_CANDIDATES_PER_RELATION = 12
MIN_RELATION_SCORE = 0.3
MIN_CANDIDATE_SCORE = 0.3
TOP_K_PATHS_FOR_REASONING = 5

CONFIG = {
    "dataset": "qa",
    "input_csv": str(INPUT_CSV),
    "input_split": INPUT_SPLIT,
    "output_dir": str(OUTPUT_DIR),
    "start_question": START_QUESTION,
    "samples": SAMPLES,
    "single_question": SINGLE_QUESTION,
    "max_length": MAX_LENGTH,
    "temperature_exploration": TEMPERATURE_EXPLORATION,
    "temperature_reasoning": TEMPERATURE_REASONING,
    "width": WIDTH,
    "depth": DEPTH,
    "min_relation_score": MIN_RELATION_SCORE,
    "min_candidate_score": MIN_CANDIDATE_SCORE,
    "top_k_paths_for_reasoning": TOP_K_PATHS_FOR_REASONING,
    "candidate_pool_size": CANDIDATE_POOL_SIZE,
    "max_candidates_per_relation": MAX_CANDIDATES_PER_RELATION,
    "LLM_type": MODEL,
    "entity_model": MODEL,
    "api_base": API_BASE,
    "opeani_api_keys": API_KEY,
    "neo4j_uri": NEO4J_URI,
    "neo4j_user": NEO4J_USER,
    "neo4j_password": NEO4J_PASSWORD,
}


def _summarize_results(results: dict[str, list[dict]]) -> None:
    dataset_names = list(results.keys())
    print(f"运行完成，数据集: {dataset_names}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    results = run_from_config(CONFIG)
    _summarize_results(results)
