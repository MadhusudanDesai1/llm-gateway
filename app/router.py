"""Cost-aware routing: cheap model -> verifier -> (maybe) strong model."""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone

from .llm import call
from .settings import CFG
from .store import get_store
from .verifier import split_confidence, verify

SYS_CHEAP = (
    "You are a helpful assistant. Answer accurately and concisely. "
    "After your answer, on a new final line, write exactly `CONFIDENCE: <number between 0 and 1>` "
    "= your probability that your answer is fully correct. Be honest; use low values when unsure."
)
SYS_STRONG = "You are a helpful assistant. Answer accurately and concisely."

store = get_store()


def run_cheap(prompt):
    res = call("cheap", [{"role": "system", "content": SYS_CHEAP}, {"role": "user", "content": prompt}])
    answer, conf = split_confidence(res.text)
    return res, answer, conf


def run_strong(prompt):
    return call("strong", [{"role": "system", "content": SYS_STRONG}, {"role": "user", "content": prompt}])


def handle(prompt: str, threshold: float | None = None) -> dict:
    thr = CFG["verifier"]["threshold"] if threshold is None else threshold
    mode = CFG["verifier"]["mode"]
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()

    c_res, answer, conf = run_cheap(prompt)
    calls = [c_res]
    verdict = verify(prompt, answer, conf, c_res.finish_reason, mode)
    if verdict.judge:
        calls.append(verdict.judge)

    escalated, degraded = verdict.score < thr, False
    strong_equiv = c_res.cost_usd * CFG["verifier"].get("strong_cost_ratio", 2.0)  # estimate; measured ~3.4x in benchmark
    if escalated:
        try:
            s_res = run_strong(prompt)
            calls.append(s_res)
            answer = s_res.text
            strong_equiv = s_res.cost_usd
        except Exception:
            escalated, degraded = False, True  # fallback: serve the cheap answer rather than fail

    latency_ms = (time.perf_counter() - t0) * 1000
    cost = sum(c.cost_usd for c in calls)
    route = "escalated" if escalated else "cheap_only"

    store.log(
        {
            "request_id": rid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "route": route,
            "models_used": [c.model for c in calls],
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "cost_usd": cost,
            "strong_equiv_cost_usd": strong_equiv,
            "latency_ms": latency_ms,
            "verifier_score": verdict.score,
            "verifier_mode": mode,
            "verifier_reason": verdict.reason,
        }
    )
    return {
        "request_id": rid,
        "answer": answer,
        "route": route,
        "escalated": escalated,
        "degraded": degraded,
        "models_used": [c.model for c in calls],
        "verifier_score": verdict.score,
        "threshold": thr,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms),
    }
