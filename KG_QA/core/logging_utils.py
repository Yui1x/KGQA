from __future__ import annotations

from typing import Any


def build_trace_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies: list[float] = []
    parse_failure_count = 0
    fallback_reasons: dict[str, int] = {}
    per_stage_token_usage: dict[str, int] = {}
    per_stage_cost_estimate: dict[str, float] = {}

    for result in results:
        for trace in result.get("llm_traces", []) or []:
            latency = trace.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

            if trace.get("parse_failure_reason"):
                parse_failure_count += 1

            fallback_reason = trace.get("fallback_reason")
            if fallback_reason:
                fallback_reasons[str(fallback_reason)] = fallback_reasons.get(str(fallback_reason), 0) + 1

            stage = str(trace.get("stage", "") or "").strip()
            usage = trace.get("usage") or {}
            if stage and isinstance(usage, dict):
                per_stage_token_usage[stage] = per_stage_token_usage.get(stage, 0) + int(usage.get("total_tokens", 0) or 0)

            if stage:
                per_stage_cost_estimate[stage] = round(
                    per_stage_cost_estimate.get(stage, 0.0) + float(trace.get("cost_estimate", 0.0) or 0.0),
                    10,
                )

    api_latency_ms = {
        "count": len(latencies),
        "average": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "max": round(max(latencies), 3) if latencies else 0.0,
    }
    return {
        "api_latency_ms": api_latency_ms,
        "parse_failure_count": parse_failure_count,
        "fallback_reasons": fallback_reasons,
        "per_stage_token_usage": per_stage_token_usage,
        "per_stage_cost_estimate": per_stage_cost_estimate,
    }
