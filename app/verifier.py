"""Verifier: decides whether the cheap model's answer is good enough."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMResult, call
from .settings import CFG

REFUSAL = re.compile(
    r"\b(i can('|’)?t (help|assist)|i cannot (help|assist)|i('m| am) unable to|as an ai (language )?model)\b",
    re.I,
)
# Matches: "CONFIDENCE: 0.95", "**Confidence:** 0.9", "Confidence: 95%", with trailing markdown/punctuation.
CONF_RE = re.compile(r"confidence\W{0,6}(\d*\.?\d+)\s*(%?)\W*$", re.I)

JUDGE_SYSTEM = """You are a strict grader. Decide whether the ANSWER correctly and fully answers the QUESTION.
Rules:
- Judge correctness ONLY. Do not reward length, confidence, tone or formatting.
- Work out the right answer yourself before grading; do not trust the ANSWER.
- Score = probability (0.0 to 1.0) that the ANSWER is correct. 1.0 = verified correct, 0.5 = cannot tell, 0.0 = wrong.
Reply with ONLY a JSON object, no other text: {"score": <number>}"""


@dataclass
class Verdict:
    score: float
    reason: str
    judge: LLMResult | None = None


def split_confidence(text: str):
    """Strip the trailing confidence line. Returns (clean_text, confidence|None)."""
    m = CONF_RE.search(text)
    if not m:
        return text.strip(), None
    val = float(m.group(1))
    if m.group(2) or val > 1:
        val = val / 100 if val <= 100 else None
    if val is not None and not (0 <= val <= 1):
        val = None
    return text[: m.start()].rstrip(" *_-\n\t"), val


def heuristics(answer: str, finish_reason: str = ""):
    if not answer.strip():
        return False, "empty"
    if finish_reason == "length":
        return False, "truncated"
    if REFUSAL.search(answer):
        return False, "refusal"
    return True, "ok"


def judge(question: str, answer: str, role: str | None = None):
    role = role or CFG["verifier"]["judge_role"]
    user = f"QUESTION:\n{question[:4000]}\n\nANSWER TO GRADE:\n{answer[:2000]}"
    res = call(role, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
    m = re.search(r'"score"\s*:\s*(\d*\.?\d+)', res.text)
    score = min(1.0, max(0.0, float(m.group(1)))) if m else 0.0  # unparseable -> escalate (fail safe)
    return score, res


def verify(question, answer, self_conf, finish_reason, mode=None) -> Verdict:
    mode = mode or CFG["verifier"]["mode"]
    ok, reason = heuristics(answer, finish_reason)
    if not ok:
        return Verdict(0.0, f"heuristic_{reason}")
    if mode == "heuristic":
        return Verdict(1.0, "heuristics_pass")
    if mode == "self":
        return Verdict(self_conf if self_conf is not None else 0.0, "self_confidence")
    if mode == "hybrid" and self_conf is not None and self_conf >= CFG["verifier"]["hybrid_gate"]:
        return Verdict(self_conf, "self_confidence_above_gate")
    try:
        score, res = judge(question, answer)
        return Verdict(score, "judge", res)
    except Exception:
        return Verdict(0.0, "judge_failed")
