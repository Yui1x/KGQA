import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_FILE = PROJECT_ROOT / "cluster" / "tail_clusters_old_to_new_merged.json"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data_1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_after_cluster"
SKIP_FILES = {
    "name_node_properties.json",
    "ord_reaction_pairs.json",
}


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping = payload.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} does not contain a valid 'mapping' object")

    cleaned: dict[str, dict[str, str]] = {}
    for file_name, file_mapping in mapping.items():
        if not isinstance(file_name, str) or not isinstance(file_mapping, dict):
            raise ValueError(f"Invalid mapping block for {file_name!r}")

        cleaned[file_name] = {}
        for old, new in file_mapping.items():
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError(f"Non-string mapping entry found in {file_name}")
            cleaned[file_name][old] = new

    return cleaned


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a JSON list")

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item #{index} is not a JSON object")
        if "tail" not in item:
            raise ValueError(f"{path} item #{index} does not contain 'tail'")

    return payload


def rewrite_file(input_path: Path, output_path: Path, file_mapping: dict[str, str]) -> tuple[int, int]:
    records = load_records(input_path)
    replaced = 0

    for item in records:
        tail = item.get("tail")
        if isinstance(tail, str) and tail in file_mapping:
            new_tail = file_mapping[tail]
            if new_tail != tail:
                replaced += 1
            item["tail"] = new_tail

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records), replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply tail cluster canonical-name mapping to extracted triple JSON files."
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help="JSON file containing the old-to-new tail mapping.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing original extracted triple JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write remapped triple JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    mapping = load_mapping(args.mapping_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_records = 0
    total_replaced = 0

    for input_path in sorted(input_dir.glob("*.json")):
        if input_path.name in SKIP_FILES:
            continue

        output_path = output_dir / input_path.name
        file_mapping = mapping.get(input_path.name, {})
        record_count, replaced_count = rewrite_file(input_path, output_path, file_mapping)

        total_files += 1
        total_records += record_count
        total_replaced += replaced_count

        print(
            f"{input_path.name}: wrote {record_count} records to {output_path} "
            f"(tails remapped: {replaced_count})"
        )

    print(f"Processed files: {total_files}")
    print(f"Processed records: {total_records}")
    print(f"Remapped tails: {total_replaced}")


if __name__ == "__main__":
    main()
