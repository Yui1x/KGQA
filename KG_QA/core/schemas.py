from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryPlan:
    raw_question: str
    qtype: str
    chemicals: list[str]
    event: str
    bridge: str
    target: str
    constraint: list[str]
    expected_slots: list[str]
    comparison_axes: list[str]
