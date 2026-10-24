"""Single place that talks to Groq (via LiteLLM): throttle, retries, token counts, cost."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import litellm

from .settings import CFG

litellm.suppress_debug_info = True

RETRYABLE = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
)


class DailyLimitError(Exception):
    """Groq daily cap (TPD/RPD) reached: retrying is pointless."""


def _is_daily(e) -> bool:
    t = str(e).lower()
    return "per day" in t or "(tpd)" in t or "(rpd)" in t


_lock = threading.Lock()
_next_slot: dict = {}


def _throttle(model_id: str):
    """Reserve a time slot so requests to one model are spaced >= 60/rpm seconds apart."""
    rpm = CFG.get("rpm")
    if not rpm:
        return
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot.get(model_id, 0.0))
        _next_slot[model_id] = start + 60.0 / rpm
    if start > now:
        time.sleep(start - now)


@dataclass
class LLMResult:
    role: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: str


def price(role: str, in_tok: int, out_tok: int) -> float:
    m = CFG["models"][role]
    return (in_tok * m["input_per_m"] + out_tok * m["output_per_m"]) / 1_000_000


def call(role, messages, max_tokens=None, temperature=0.0, retries=10) -> LLMResult:
    m = CFG["models"][role]
    kwargs = dict(
        model=f"groq/{m['id']}",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or m["max_tokens"],
        timeout=60,
    )
    if m.get("extra"):
        kwargs["extra_body"] = dict(m["extra"])

    for attempt in range(retries):
        _throttle(m["id"])
        t0 = time.perf_counter()  # latency = successful attempt only
        try:
            r = litellm.completion(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000
            break
        except RETRYABLE as e:
            if _is_daily(e):
                raise DailyLimitError(f"{m['id']}: {str(e)[-220:]}") from e
            if attempt == retries - 1:
                raise
            wait = min(2 ** attempt, 30) + random.random()
            print(f"[retry] {m['id']} attempt {attempt + 1}: {type(e).__name__} -> sleep {wait:.0f}s", flush=True)
            time.sleep(wait)

    in_tok = r.usage.prompt_tokens or 0
    out_tok = r.usage.completion_tokens or 0  # includes reasoning tokens
    return LLMResult(
        role=role,
        model=m["id"],
        text=(r.choices[0].message.content or "").strip(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=price(role, in_tok, out_tok),
        latency_ms=latency_ms,
        finish_reason=r.choices[0].finish_reason or "",
    )
