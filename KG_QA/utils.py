from __future__ import annotations

import csv
import json
import time
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMUsageStats:
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    traces: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        return {
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


_ACTIVE_LLM_USAGE: ContextVar[LLMUsageStats | None] = ContextVar("active_llm_usage", default=None)


@contextmanager
def track_llm_usage():
    stats = LLMUsageStats()
    token = _ACTIVE_LLM_USAGE.set(stats)
    try:
        yield stats
    finally:
        _ACTIVE_LLM_USAGE.reset(token)


def _record_llm_request(
    usage: dict[str, Any] | None = None,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    latency_ms: float = 0.0,
) -> None:
    stats = _ACTIVE_LLM_USAGE.get()
    if stats is None:
        return

    stats.request_count += 1
    usage = usage or {}
    stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
    stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
    stats.total_tokens += int(usage.get("total_tokens", 0) or 0)
    stats.traces.append(
        {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "latency_ms": round(latency_ms, 3),
            "cost_estimate": 0.0,
        }
    )


def build_chat_completion_url(api_base: str) -> str:
    cleaned = api_base.rstrip("/")
    if not cleaned:
        raise ValueError("api_base is required. Set KGQA_API_BASE or pass --api_base.")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def call_chat_model(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 120,
) -> str:
    url = build_chat_completion_url(api_base)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started_at = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    latency_ms = (time.perf_counter() - started_at) * 1000
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response does not contain choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        _record_llm_request(
            data.get("usage") if isinstance(data, dict) else None,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=text,
            latency_ms=latency_ms,
        )
        return text
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if item.get("type") == "text").strip()
        if text:
            _record_llm_request(
                data.get("usage") if isinstance(data, dict) else None,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=text,
                latency_ms=latency_ms,
            )
            return text
    raise ValueError("LLM response does not contain usable content.")


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def call_chat_json(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 120,
) -> dict[str, Any] | None:
    response = call_chat_model(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return extract_json_object(response)


def normalize_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def overlap_score(query: str, text: str) -> float:
    query_chars = set(normalize_text(query))
    text_chars = set(normalize_text(text))
    if not query_chars or not text_chars:
        return 0.0
    overlap = len(query_chars & text_chars)
    return overlap / max(len(query_chars), 1)


def render_path_chain(triplets: list[str]) -> str:
    if not triplets:
        return ""
    chain = str(triplets[0]).strip()
    pattern = re.compile(r"^\s*(.*?)\s*-\[(.*?)\]->\s*(.*?)\s*$")
    for triplet in triplets[1:]:
        match = pattern.match(str(triplet))
        if not match:
            return " | ".join(str(item).strip() for item in triplets if str(item).strip())
        _, relation, tail = match.groups()
        chain += f" -[{relation}]-> {tail}"
    return chain


def dump_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    fieldnames = [
        "题目",
        "参考答案",
        "LLM答案",
        "LLM依据",
        "依据路径",
        "消耗时间",
        "LLM请求次数",
        "消耗总token",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_qa_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            question = (row.get("Questions") or "").strip()
            answer = (row.get("Answers") or "").strip()
            if question:
                rows.append({"question": question, "reference_answer": answer})
    return rows


def slice_rows(rows: list[dict[str, str]], start_question: int, samples: int) -> list[dict[str, str]]:
    start_index = max(start_question - 1, 0)
    end_index = min(start_index + samples, len(rows))
    return rows[start_index:end_index]
