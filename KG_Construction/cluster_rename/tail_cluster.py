import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import AgglomerativeClustering


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_FILES = {
    "name_node_properties.json",
    "merge_stats.json",
    "disposal_transport_regulation_regulation_info_tail_dedup.json",
}


def iter_source_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.json")):
        if path.name not in EXCLUDED_FILES:
            yield path


def load_unique_tails(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")

    unique_tails: list[str] = []
    seen: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        tail = item.get("tail")
        if not isinstance(tail, str):
            continue
        tail = tail.strip()
        if not tail or tail in seen:
            continue
        seen.add(tail)
        unique_tails.append(tail)

    return unique_tails


def get_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> list[int]:
    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1:
        return [0]

    distance_matrix = 1.0 - np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    return clustering.fit_predict(distance_matrix).tolist()


def group_texts_by_labels(texts: list[str], labels: list[int]) -> list[list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for text, label in zip(texts, labels):
        grouped[int(label)].append(text)
    return [grouped[label] for label in sorted(grouped)]


def build_cluster_payload(
    file_name: str,
    tails: list[str],
    clusters: list[list[str]],
    threshold: float,
    model_name: str,
) -> dict:
    return {
        "file": file_name,
        "model": model_name,
        "distance_threshold": threshold,
        "unique_tail_count": len(tails),
        "cluster_count": len(clusters),
        "singleton_cluster_count": sum(1 for cluster in clusters if len(cluster) == 1),
        "clusters": clusters,
    }


def sanitize_output_name(file_name: str) -> str:
    return file_name.replace(".json", "") + "_tail_clusters.json"


def cluster_file(
    path: Path,
    model_name: str,
    batch_size: int,
    device: str,
    distance_threshold: float,
) -> dict:
    tails = load_unique_tails(path)
    if not tails:
        return build_cluster_payload(
            file_name=path.name,
            tails=[],
            clusters=[],
            threshold=distance_threshold,
            model_name=model_name,
        )

    embeddings = encode_texts(
        texts=tails,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    labels = cluster_embeddings(embeddings, distance_threshold=distance_threshold)
    clusters = group_texts_by_labels(tails, labels)
    return build_cluster_payload(
        file_name=path.name,
        tails=tails,
        clusters=clusters,
        threshold=distance_threshold,
        model_name=model_name,
    )


def write_outputs(output_dir: Path, payloads: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "tail_clusters_all.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump({"files": payloads}, f, ensure_ascii=False, indent=2)

    for payload in payloads:
        file_path = output_dir / sanitize_output_name(payload["file"])
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster tail texts in JSON files with BGE embeddings and AgglomerativeClustering."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data_1",
        help="Directory containing source JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tail_cluster_results",
        help="Directory to save clustering outputs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-large-zh-v1.5",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device selection.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.18,
        help="Agglomerative clustering distance threshold on cosine distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    device = get_device(args.device)
    payloads: list[dict] = []

    for path in iter_source_files(args.data_dir):
        print(f"Clustering {path.name} on {device} ...")
        payload = cluster_file(
            path=path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=device,
            distance_threshold=args.distance_threshold,
        )
        payloads.append(payload)
        print(
            f"  unique tails: {payload['unique_tail_count']}, "
            f"clusters: {payload['cluster_count']}"
        )

    write_outputs(args.output_dir, payloads)
    print(f"Saved clustering outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
