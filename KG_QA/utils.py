from __future__ import annotations

import csv
import json
import random
import threading
import time
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

# One pooled session shared by all worker threads.
#
# requests.post() creates a fresh connection (and, over HTTPS, a fresh TLS
# handshake) per call. At hundreds of concurrent questions issuing 10-30 calls
# each, that setup dominates wall-clock time well before the API does. A shared
# session with a large pool keeps connections alive across calls.
#
# pool_maxsize must be >= the worker count, otherwise urllib3 opens and discards
# connections, which is the same problem in disguise.
_SESSION = requests.Session()
_HTTP_POOL_SIZE = 512
_SESSION.mount(
    "http://",
    HTTPAdapter(pool_connections=64, pool_maxsize=_HTTP_POOL_SIZE, max_retries=0),
)
_SESSION.mount(
    "https",
    HTTPAdapter(pool_connections=64, pool_maxsize=_HTTP_POOL_SIZE, max_retries=0),
)


def configure_http_pool(size: int) -> None:
    """Grow the connection pool to match the worker count."""
    global _HTTP_POOL_SIZE
    size = max(16, int(size))
    _HTTP_POOL_SIZE = size
    for prefix in ("http://", "https://"):
        _SESSION.mount(
            prefix,
            HTTPAdapter(pool_connections=min(64, size), pool_maxsize=size, max_retries=0),
        )

# Transport-level retry policy. Tuned for gateways that shed load with 503
# rather than queueing; see the comment inside call_chat_model.
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 15.0

# 429 backoff is longer than a transient-error backoff: the upstream quota needs
# time to recover, and retrying fast only burns the budget again.
RATE_LIMIT_BASE_DELAY = 4.0
RATE_LIMIT_MAX_DELAY = 45.0


# ------------------------------------------------------------------ client pool ---
#
# The gateway enforces a per-account limit ("more than 120 requests in 60s") and
# each account tolerates ~2 concurrent requests. One question costs 25-35 calls,
# so the ceiling is roughly 4 questions/minute no matter how many threads are
# started. Two mechanisms keep the client inside that ceiling instead of
# discovering it through 429s:
#
#   1. KeyPool     -- round-robin over N API keys so each account sees ~1/N of
#                     the traffic.
#   2. RateLimiter -- a shared token bucket that caps total requests/minute
#                     across every thread and every key.
#
# Both are process-global and created lazily from call_chat_model, so callers
# that construct their own SimpleNamespace (the offline scripts) still get the
# same behaviour as the runner.

_API_KEYS: list[str] = []
_API_KEY_CURSOR = 0
_KEY_LOCK = threading.Lock()

_RATE_LOCK = threading.Lock()
_RATE_TIMES: list[float] = []
_RATE_LIMIT_PER_MIN = 0          # 0 disables pacing

# Adaptive concurrency control.
#
# The upstream quota is not documented and shifts between accounts, so a fixed
# request ceiling cannot be right all the time. Instead of guessing, the client
# watches for 429s and adjusts a *token bucket of in-flight slots*:
#
#   * every successful request slowly raises the allowance (additive increase)
#   * every 429 cuts it multiplicatively and drains the current window
#
# This is AIMD, the same control loop TCP uses. It converges to whatever the
# real ceiling is without anyone having to know it.
#
# `--concurrency` then becomes an upper bound on worker threads, not on requests:
# N workers still issue at most as many requests as the limiter permits.
_LIMIT_LOCK = threading.Lock()
_inflight_limit = 64             # current AIMD allowance (requests per window)
_inflight_limit_max = 64         # ceiling set from --concurrency
_inflight_sem: threading.Semaphore | None = None
_consecutive_ok = 0


def configure_adaptive_limit(max_inflight: int) -> None:
    """Enable AIMD throttling with `max_inflight` as the hard ceiling."""
    global _inflight_limit, _inflight_limit_max, _inflight_sem, _consecutive_ok
    with _LIMIT_LOCK:
        _inflight_limit_max = max(1, int(max_inflight))
        _inflight_limit = _inflight_limit_max
        _inflight_sem = threading.BoundedSemaphore(_inflight_limit)
        _consecutive_ok = 0


def _acquire_slot() -> None:
    sem = _inflight_sem
    if sem is not None:
        sem.acquire()


def _release_slot(success: bool) -> None:
    """Return the slot, shrinking or growing the allowance based on the result."""
    global _inflight_limit, _consecutive_ok, _inflight_sem
    with _LIMIT_LOCK:
        if success:
            _consecutive_ok += 1
            # Additive increase: one extra slot per 25 consecutive successes,
            # capped. Slow enough that a bursty upstream does not oscillate.
            if _consecutive_ok >= 25 and _inflight_limit < _inflight_limit_max:
                _inflight_limit += 1
                _consecutive_ok = 0
                _grow_semaphore(_inflight_limit)
        else:
            _consecutive_ok = 0
            # Multiplicative decrease on 429: halve the allowance, floor at 2.
            new_limit = max(2, _inflight_limit // 2)
            if new_limit < _inflight_limit:
                _inflight_limit = new_limit
                _grow_semaphore(_inflight_limit)
    sem = _inflight_sem
    if sem is not None:
        try:
            sem.release()
        except ValueError:
            pass


def _grow_semaphore(new_limit: int) -> None:
    """Replace the semaphore with one of a different size.

    A semaphore cannot be resized, so on a shrink we hand out permits to drain
    the surplus waiters. Only called with _LIMIT_LOCK held.
    """
    global _inflight_sem
    _inflight_sem = threading.BoundedSemaphore(new_limit)


def current_limit() -> dict[str, int]:
    with _LIMIT_LOCK:
        return {"inflight_limit": _inflight_limit,
                "inflight_max": _inflight_limit_max,
                "consecutive_ok": _consecutive_ok}


def configure_client(api_keys: list[str] | None = None, requests_per_minute: int = 0) -> None:
    """Set the key pool and an optional fixed request ceiling.

    `requests_per_minute` is a fallback for gateways with a documented fixed
    limit. Leave it at 0 and call `configure_adaptive_limit()` instead when the
    quota is unknown.
    """
    global _API_KEYS, _API_KEY_CURSOR, _RATE_LIMIT_PER_MIN
    with _KEY_LOCK:
        keys = [k.strip() for k in (api_keys or []) if k and k.strip()]
        if keys:
            _API_KEYS = keys
            _API_KEY_CURSOR = 0
    with _RATE_LOCK:
        _RATE_LIMIT_PER_MIN = max(0, int(requests_per_minute))
        _RATE_TIMES.clear()


def next_api_key(fallback: str = "") -> str:
    global _API_KEY_CURSOR
    with _KEY_LOCK:
        if not _API_KEYS:
            return fallback
        key = _API_KEYS[_API_KEY_CURSOR % len(_API_KEYS)]
        _API_KEY_CURSOR += 1
        return key


def _throttle() -> None:
    """Space requests evenly without ever sleeping inside the lock.

    Two earlier versions were wrong in opposite directions:

      1. Reserve a future timestamp and return immediately. Legal average rate,
         but every waiting thread woke at the same instant once the window
         rolled over -- a burst that trips the gateway's own limiter.
      2. Sleep while holding the lock. Correct pacing, but the lock is held for
         the whole interval, so 20 worker threads serialise to one request per
         interval and the runner appears hung (observed: 0 questions in 200s).

    This version computes the wait under the lock, releases it, then sleeps.
    The timestamp is recorded before sleeping, so concurrent callers see it and
    queue behind each other rather than stacking up.
    """
    if _RATE_LIMIT_PER_MIN <= 0:
        return
    window = 60.0
    min_interval = window / _RATE_LIMIT_PER_MIN
    with _RATE_LOCK:
        now = time.monotonic()
        cutoff = now - window
        while _RATE_TIMES and _RATE_TIMES[0] < cutoff:
            _RATE_TIMES.pop(0)
        if _RATE_TIMES and len(_RATE_TIMES) >= _RATE_LIMIT_PER_MIN:
            wait = window - (now - _RATE_TIMES[0])
        elif _RATE_TIMES:
            wait = max(min_interval - (now - _RATE_TIMES[-1]), 0.0)
        else:
            wait = 0.0
        # Record the slot we are about to occupy so the next caller measures
        # its gap from here, not from the moment this one wakes.
        _RATE_TIMES.append(now + wait)
    if wait > 0:
        time.sleep(wait)


def client_status() -> dict[str, Any]:
    with _KEY_LOCK:
        n_keys = len(_API_KEYS)
    with _RATE_LOCK:
        used = len(_RATE_TIMES)
    return {"keys": n_keys, "requests_in_window": used,
            "limit_per_min": _RATE_LIMIT_PER_MIN}



@dataclass
class LLMUsageStats:
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    traces: list[dict[str, str]] = field(default_factory=list)
    # finish_reason per completed request ("stop", "length", ...). A "length"
    # count above zero means some call was truncated and its JSON is unusable.
    finish_reasons: dict[str, int] = field(default_factory=dict)
    # Every place where the pipeline substitutes a non-LLM decision for an LLM
    # one increments a named counter here, so a run can prove it did not
    # silently degrade (see record_fallback).
    fallbacks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reasons": dict(self.finish_reasons),
            "fallbacks": dict(self.fallbacks),
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


def record_fallback(kind: str, amount: int = 1) -> None:
    """Count a non-LLM substitution (e.g. lexical ranking after an API error)."""
    stats = _ACTIVE_LLM_USAGE.get()
    if stats is None:
        return
    stats.fallbacks[kind] = stats.fallbacks.get(kind, 0) + amount


def _record_llm_request(
    usage: dict[str, Any] | None = None,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    latency_ms: float = 0.0,
    finish_reason: str = "",
) -> None:
    stats = _ACTIVE_LLM_USAGE.get()
    if stats is None:
        return

    reason = str(finish_reason or "unknown")
    stats.finish_reasons[reason] = stats.finish_reasons.get(reason, 0) + 1
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
            "finish_reason": reason,
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


# Which backends actually honour reasoning_effort="none".
#
# Measured on this gateway (one 64-token JSON probe per model, thinking tokens
# read from usage.completion_thinking_tokens):
#
#   deepseek-v4.1-flash  none -> think=0,  content emitted   SUPPORTED
#   mimo-v2.6-pro        none -> think=0,  content emitted   SUPPORTED
#   kimi-k2.5            none -> think=1,  content emitted   SUPPORTED
#   glm-5.3-flash        none -> think=61 of 64, EMPTY       BREAKS
#   glm-5.3              none -> think=62 of 64, EMPTY       BREAKS
#
# For the glm family the parameter is worse than useless: the model spends the
# whole budget on hidden reasoning and returns an empty string, which downstream
# reads as "no slots" and degrades to abstention. So a request to disable
# reasoning is only honoured for models on this list; for anything else the
# parameter is dropped and the backend default applies.
REASONING_DISABLE_SAFE = frozenset({
    "cn:deepseek-v4.1-flash",
    "cn:deepseek-v4-flash",
    "cn:deepseek-v4-pro",
    "cn:mimo-v2.6-pro",
    "cn:kimi-k2.5",
    # Official DeepSeek API ids (no cn: prefix). Verified: reasoning_effort="none"
    # drops reasoning_tokens to zero and still emits content.
    "deepseek-flash",
    "deepseek-chat",
})


def resolve_reasoning(model: str, reasoning: str) -> str | None:
    """Return the reasoning_effort to send, or None to omit the parameter."""
    if reasoning == "auto":
        return None
    if reasoning == "none" and model not in REASONING_DISABLE_SAFE:
        return None
    return reasoning


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
    reasoning: str = "auto",
) -> str:
    """Send one chat completion.

    `reasoning` controls the hidden chain-of-thought budget:
      "auto"  -- leave the backend default (reasoning on for thinking models)
      "none"  -- disable reasoning entirely, for schema-constrained calls whose
                 output is a short JSON object. Verified on this gateway:
                 reasoning_effort="none" yields completion_thinking_tokens=0,
                 which roughly halves completion cost and latency.
      "high" / "medium" / "low" -- explicit effort, for judgement calls where
                 the reasoning trace is doing real work.
    """
    url = build_chat_completion_url(api_base)
    headers = {"Content-Type": "application/json"}
    key = next_api_key(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning != "auto":
        effort = resolve_reasoning(model, reasoning)
        if effort is not None:
            payload["reasoning_effort"] = effort
    # Per-request retry with jittered exponential backoff.
    #
    # A single question issues 25-35 chat completions. Hosted gateways fail a
    # small fraction of requests under load (observed: 503 no_healthy_account
    # above ~8 concurrent, plus transient resets). Without a retry here, a 4%
    # per-request failure rate compounds into 0.96**30 ~= 29% of questions
    # producing no answer at all -- and because the callers swallow exceptions,
    # such failures surface as "abstained" rather than as errors.
    #
    # Retries cover connection errors, timeouts, 429, and 5xx. Any other 4xx is
    # a genuine request problem and is raised immediately.
    retriable_status = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    last_error: Exception | None = None
    response = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        _throttle()
        _acquire_slot()
        succeeded = False
        started_at = time.perf_counter()
        try:
            response = _SESSION.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in retriable_status:
                raise requests.HTTPError(
                    f"{response.status_code} from gateway: {response.text[:200]}",
                    response=response,
                )
            response.raise_for_status()
            succeeded = True
            break
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in retriable_status:
                raise
            if attempt >= RETRY_ATTEMPTS:
                raise
            # A 429 means we are above the real ceiling. Back off longer than for
            # a transient network error, and let the AIMD controller shrink the
            # allowance so other threads stop piling on.
            if status == 429:
                delay = min(RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1)), RATE_LIMIT_MAX_DELAY)
                delay *= 1.0 + random.random()
            else:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                delay *= 0.5 + random.random()
            time.sleep(delay)
        finally:
            _release_slot(succeeded)
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError(f"LLM request failed after {RETRY_ATTEMPTS} attempts") from last_error

    data = response.json()
    latency_ms = (time.perf_counter() - started_at) * 1000
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response does not contain choices.")
    message = choices[0].get("message") or {}
    finish_reason = str(choices[0].get("finish_reason") or "")
    usage_block = data.get("usage") if isinstance(data, dict) else None
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        _record_llm_request(
            usage_block,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=text,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
        return text
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if item.get("type") == "text").strip()
        if text:
            _record_llm_request(
                usage_block,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=text,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
            return text
    # The request was billed even though nothing usable came back (typically a
    # reasoning model that spent max_tokens on hidden reasoning). Record it so
    # token counts and finish_reason="length" are visible, then fail loudly.
    _record_llm_request(
        usage_block,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_text="",
        latency_ms=latency_ms,
        finish_reason=finish_reason or "no_content",
    )
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
    reasoning: str = "auto",
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
        reasoning=reasoning,
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
