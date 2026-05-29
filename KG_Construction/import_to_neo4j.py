from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_after_cluster"
CHEMICAL_LABEL = "Chemical"


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().replace(" ", "_")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
        raise ValueError(f"Unsupported label name: {raw_label}")
    return label


def load_json(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_chemical_names(data_dir: Path) -> set[str]:
    name_file = data_dir / "name_node_properties.json"
    return {
        str(row.get("中文名称") or "").strip()
        for row in load_json(name_file)
        if str(row.get("中文名称") or "").strip()
    }


def relation_file_sort_key(file_path: Path) -> tuple[int, str]:
    if file_path.stem == "gb30000":
        return (1, file_path.name)
    return (0, file_path.name)


def build_relation_props(record: dict) -> dict:
    props = {}
    for key in ("evidence", "reference"):
        value = record.get(key)
        if value not in (None, ""):
            props[key] = value
    return props


def run_batch_query(driver, query: str, rows: list[dict], batch_size: int = 2000) -> int:
    imported_count = 0
    with driver.session() as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            result = session.execute_write(
                lambda tx, payload=batch: tx.run(query, rows=payload).single()["imported_count"]
            )
            imported_count += result
    return imported_count


def ensure_constraints(driver, relation_files: list[Path]) -> None:
    with driver.session() as session:
        session.run(
            """
            CREATE CONSTRAINT chemical_chinese_name_unique IF NOT EXISTS
            FOR (chemical:Chemical) REQUIRE chemical.`中文名称` IS UNIQUE
            """
        ).consume()

        for file_path in relation_files:
            label = normalize_label(file_path.stem)
            constraint_name = f"{label}_name_unique"
            session.run(
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (node:`{label}`) REQUIRE node.name IS UNIQUE"
            ).consume()


def clear_database(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()


def import_chemical_properties(driver, data_dir: Path) -> int:
    rows = [
        row
        for row in load_json(data_dir / "name_node_properties.json")
        if str(row.get("中文名称") or "").strip()
    ]
    query = """
    UNWIND $rows AS row
    MERGE (chemical:Chemical {`中文名称`: row.`中文名称`})
    SET chemical += row
    RETURN count(chemical) AS imported_count
    """
    return run_batch_query(driver, query, rows)


def import_relation_file(
    driver,
    file_path: Path,
    chemical_names: set[str],
) -> int:
    tail_label = normalize_label(file_path.stem)
    raw_rows = load_json(file_path)
    chemical_tail_rows = []
    other_tail_rows = []
    for record in raw_rows:
        row = {
            "head": record["head"],
            "tail": record["tail"],
            "tail_label": tail_label,
            "relation_type": record.get("relation", tail_label),
            "identity_props": build_relation_props(record),
            "set_props": build_relation_props(record),
        }
        if row["tail"] in chemical_names:
            chemical_tail_rows.append(row)
        else:
            other_tail_rows.append(row)

    chemical_tail_query = """
    UNWIND $rows AS row
    MATCH (head:Chemical {`中文名称`: row.head})
    MATCH (tail:Chemical {`中文名称`: row.tail})
    CALL apoc.merge.relationship(
        head,
        row.relation_type,
        row.identity_props,
        row.set_props,
        tail,
        row.set_props
    ) YIELD rel
    RETURN count(rel) AS imported_count
    """

    other_tail_query = """
    UNWIND $rows AS row
    MATCH (head:Chemical {`中文名称`: row.head})
    CALL apoc.merge.node([row.tail_label], {name: row.tail}, {}, {}) YIELD node AS tail
    CALL apoc.merge.relationship(
        head,
        row.relation_type,
        row.identity_props,
        row.set_props,
        tail,
        row.set_props
    ) YIELD rel
    RETURN count(rel) AS imported_count
    """

    imported_count = 0
    if chemical_tail_rows:
        imported_count += run_batch_query(driver, chemical_tail_query, chemical_tail_rows)
    if other_tail_rows:
        imported_count += run_batch_query(driver, other_tail_query, other_tail_rows)
    return imported_count


def import_gb30000_file(driver, file_path: Path) -> int:
    rows = []
    for record in load_json(file_path):
        head = str(record.get("head") or "").strip()
        tail = str(record.get("tail") or "").strip()
        relation_type = str(record.get("relation") or "").strip()
        if not head or not tail or not relation_type:
            continue
        props = build_relation_props(record)
        rows.append(
            {
                "head": head,
                "tail": tail,
                "relation_type": relation_type,
                "identity_props": props,
                "set_props": props,
            }
        )

    query = """
    UNWIND $rows AS row
    MATCH (head:`hazard_category` {name: row.head})
    CALL apoc.merge.node(["gb30000"], {name: row.tail}, {}, {}) YIELD node AS tail
    CALL apoc.merge.relationship(
        head,
        row.relation_type,
        row.identity_props,
        row.set_props,
        tail,
        row.set_props
    ) YIELD rel
    RETURN count(rel) AS imported_count
    """
    return run_batch_query(driver, query, rows)


def import_all_data(driver, data_dir: Path, clear: bool = False) -> dict[str, int]:
    if clear:
        clear_database(driver)

    relation_files = sorted(
        (
            file_path
            for file_path in data_dir.glob("*.json")
            if file_path.name != "name_node_properties.json"
        ),
        key=relation_file_sort_key,
    )
    chemical_names = load_chemical_names(data_dir)

    ensure_constraints(driver, relation_files)
    summary: dict[str, int] = {}

    chemical_count = import_chemical_properties(driver, data_dir)
    summary["chemical_properties"] = chemical_count
    print(f"[done] name_node_properties.json -> imported {chemical_count} Chemical nodes")

    total_files = len(relation_files)
    for index, file_path in enumerate(relation_files, start=1):
        if file_path.stem == "gb30000":
            imported_count = import_gb30000_file(driver, file_path)
        else:
            imported_count = import_relation_file(driver, file_path, chemical_names)
        summary[file_path.stem] = imported_count
        print(f"[{index}/{total_files}] {file_path.name} -> imported {imported_count} relationships")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import hazardous chemical KG JSON files into Neo4j.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--neo4j-uri", default=os.getenv("KGQA_NEO4J_URI", os.getenv("NEO4J_URI", "")))
    parser.add_argument("--neo4j-user", default=os.getenv("KGQA_NEO4J_USER", os.getenv("NEO4J_USER", "")))
    parser.add_argument("--neo4j-password", default=os.getenv("KGQA_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "")))
    parser.add_argument("--clear", action="store_true", help="Delete all existing Neo4j nodes before import.")
    return parser


def main() -> dict[str, int]:
    args = build_parser().parse_args()
    if not args.neo4j_uri:
        raise ValueError("Neo4j URI is required. Set KGQA_NEO4J_URI or pass --neo4j-uri.")
    if not args.neo4j_user:
        raise ValueError("Neo4j user is required. Set KGQA_NEO4J_USER or pass --neo4j-user.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        summary = import_all_data(driver, args.data_dir, clear=args.clear)
    finally:
        driver.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
