from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from utils import normalize_text


NAME_FIELDS = ("chinese_name", "chinese_alias", "english_name", "english_alias")
EXCLUDED_ATTRIBUTE_FIELDS = {"reference"}
CHEMICAL_DB_FIELD_MAP = {
    "chinese_name": "中文名称",
    "chinese_alias": "中文别名",
    "english_name": "英文名称",
    "english_alias": "英文别名",
}


def _split_name_field(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return parts
    text = str(value or "").strip()
    if not text:
        return []
    normalized = text.replace("；", ";")
    parts = [part.strip() for part in normalized.split(";")]
    return [part for part in parts if part]


class HazardNeo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._name_records_cache: tuple[dict[str, str], ...] | None = None

    def close(self) -> None:
        self.driver.close()

    def list_name_records(self) -> tuple[dict[str, str], ...]:
        cached = getattr(self, "_name_records_cache", None)
        if cached is not None:
            return cached

        query = """
        MATCH (c:Chemical)
        RETURN
            c.`中文名称` AS chinese_name,
            c.`中文别名` AS chinese_alias,
            c.`英文名称` AS english_name,
            c.`英文别名` AS english_alias
        ORDER BY chinese_name
        """
        with self.driver.session() as session:
            rows = session.run(query).data()

        records = tuple(
            {
                field: str(row.get(field, "") or "").strip()
                for field in NAME_FIELDS
            }
            for row in rows
            if str(row.get("chinese_name", "") or "").strip()
        )
        self._name_records_cache = records
        return records

    def get_chemical_node(self, chemical_name: str) -> dict[str, Any] | None:
        query = """
        MATCH (c:Chemical {`中文名称`:$chemical_name})
        RETURN elementId(c) AS element_id, 'Chemical' AS label, c.`中文名称` AS name, properties(c) AS props
        LIMIT 1
        """
        with self.driver.session() as session:
            row = session.run(query, chemical_name=chemical_name).data()
        return row[0] if row else None

    def list_chemical_names(self) -> tuple[str, ...]:
        records = self.list_name_records()
        if records:
            names = [str(record.get("chinese_name", "")).strip() for record in records if str(record.get("chinese_name", "")).strip()]
            return tuple(sorted(set(names), key=len, reverse=True))
        query = "MATCH (c:Chemical) RETURN c.`中文名称` AS name ORDER BY name"
        with self.driver.session() as session:
            rows = session.run(query).data()
        return tuple(str(row["name"]).strip() for row in rows if str(row.get("name", "")).strip())

    def find_exact_chemical_mentions(self, question: str) -> list[str]:
        matches: list[tuple[int, int, str]] = []
        for name in self.list_chemical_names():
            if not name:
                continue
            start = question.find(name)
            while start != -1:
                matches.append((start, start + len(name), name))
                start = question.find(name, start + 1)

        if not matches:
            return []

        selected: list[tuple[int, int, str]] = []
        for start, end, name in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
            overlaps = any(not (end <= kept_start or start >= kept_end) for kept_start, kept_end, _ in selected)
            if overlaps:
                continue
            selected.append((start, end, name))

        selected.sort(key=lambda item: item[0])
        return [name for _, _, name in selected]

    def resolve_chemical_names(self, raw_names: list[str]) -> list[str]:
        records = self.list_name_records()
        normalized_map: dict[str, str] = {}
        variant_pairs: list[tuple[str, str]] = []
        if records:
            for record in records:
                canonical = str(record.get("chinese_name", "")).strip()
                if not canonical:
                    continue
                for field in NAME_FIELDS:
                    for variant in _split_name_field(record.get(field, "")):
                        normalized_variant = normalize_text(variant)
                        if not normalized_variant:
                            continue
                        normalized_map[normalized_variant] = canonical
                        variant_pairs.append((variant, canonical))
        else:
            all_names = self.list_chemical_names()
            normalized_map = {normalize_text(name): name for name in all_names}
            variant_pairs = [(name, name) for name in all_names]

        resolved = []
        for raw_name in raw_names:
            cleaned = raw_name.strip()
            if not cleaned:
                continue
            exact = normalized_map.get(normalize_text(cleaned))
            if exact and exact not in resolved:
                resolved.append(exact)
                continue
            substring_matches = [
                canonical
                for variant, canonical in variant_pairs
                if cleaned in variant or variant in cleaned
            ]
            if substring_matches:
                best = min(substring_matches, key=lambda item: abs(len(item) - len(cleaned)))
                if best not in resolved:
                    resolved.append(best)
        return resolved

    def local_entity_fallback(self, question: str) -> list[str]:
        records = self.list_name_records()
        matches: list[tuple[int, str]] = []
        if records:
            for record in records:
                canonical = str(record.get("chinese_name", "")).strip()
                if not canonical:
                    continue
                for field in ("chinese_name", "chinese_alias"):
                    for variant in _split_name_field(record.get(field, "")):
                        if variant and variant in question:
                            matches.append((question.find(variant), canonical))
        else:
            for name in self.list_chemical_names():
                if name and name in question:
                    matches.append((question.find(name), name))
        unique = []
        for _, name in sorted(matches, key=lambda item: (item[0], -len(item[1]))):
            if name in unique:
                continue
            unique.append(name)
        return unique

    def _get_node_properties(self, element_id: str) -> dict[str, Any]:
        query = """
        MATCH (n)
        WHERE elementId(n) = $element_id
        RETURN properties(n) AS props
        LIMIT 1
        """
        with self.driver.session() as session:
            rows = session.run(query, element_id=element_id).data()
        if not rows:
            return {}
        return rows[0].get("props") or {}

    def get_available_relations(self, element_id: str) -> list[dict[str, Any]]:
        query = """
        MATCH (n)-[r]->()
        WHERE elementId(n) = $element_id
        RETURN type(r) AS relation, count(*) AS freq
        ORDER BY freq DESC, relation
        """
        with self.driver.session() as session:
            rows = session.run(query, element_id=element_id).data()
        return [
            {
                "relation": row["relation"],
                "freq": int(row["freq"]),
                "source_type": "relation",
            }
            for row in rows
        ]

    def get_available_attributes(self, element_id: str) -> list[dict[str, Any]]:
        props = self._get_node_properties(element_id)
        rows = []
        for key, value in props.items():
            if key in EXCLUDED_ATTRIBUTE_FIELDS or value in (None, ""):
                continue
            rows.append(
                {
                    "relation": f"属性:{key}",
                    "freq": 1,
                    "source_type": "attribute",
                    "attribute_name": key,
                }
            )
        rows.sort(key=lambda item: item["relation"])
        return rows

    def get_available_descriptors(self, element_id: str) -> list[dict[str, Any]]:
        rows = self.get_available_relations(element_id) + self.get_available_attributes(element_id)
        rows.sort(key=lambda item: (-int(item.get("freq", 1)), item["relation"]))
        return rows

    def expand_relation(self, element_id: str, relation: str, limit: int) -> list[dict[str, Any]]:
        query = """
        MATCH (n)-[r]->(t)
        WHERE elementId(n) = $element_id AND type(r) = $relation
        RETURN
            elementId(t) AS element_id,
            head(labels(t)) AS label,
            coalesce(t.name, t.`中文名称`, t.`英文名称`, t.chinese_name, t.english_name, '') AS name,
            r.reference AS reference
        LIMIT $limit
        """
        with self.driver.session() as session:
            rows = session.run(query, element_id=element_id, relation=relation, limit=limit).data()
        return rows

    def expand_attribute(self, element_id: str, attribute_name: str) -> list[dict[str, Any]]:
        props = self._get_node_properties(element_id)
        value = props.get(attribute_name)
        if value in (None, ""):
            return []
        return [
            {
                "element_id": None,
                "label": "AttributeValue",
                "name": str(value).strip(),
                "reference": "",
                "source_type": "attribute",
            }
        ]

    def expand_descriptor(self, element_id: str, descriptor_name: str, limit: int) -> list[dict[str, Any]]:
        if descriptor_name.startswith("属性:"):
            attribute_name = descriptor_name.split(":", 1)[1].strip()
            return self.expand_attribute(element_id, attribute_name)
        return self.expand_relation(element_id, descriptor_name, limit)
