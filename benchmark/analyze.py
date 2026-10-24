"""Offline simulation of every verifier mode x judge x threshold from cached raw results.
Router cost = cheap call + judge call (when it runs) + strong call (when it escalates)."""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.settings import CFG  # noqa: E402
from app.store import pct  # noqa: E402

rows = [json.loads(line) for line in open("results/raw.jsonl")]
_by_idx = {r["idx"]: r for r in rows}
for _f in Path("results").glob("judges_*.jsonl"):
    for _line in open(_f):
        _j = json.loads(_line)
        if _j["idx"] in _by_idx:
            _by_idx[_j["idx"]].setdefault("judges", {})[_j["role"]] = _j["res"]
N = len(rows)
GATE = CFG["verifier"]["hybrid_gate"]
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
JUDGES = [j for j in CFG["benchmark"]["judge_roles"]
          if all(j in r.get("judges", {}) for r in rows if r["heur_ok"])]
COMBOS = [("heuristic", None), ("self", None)] + [(m, j) for j in JUDGES for m in ("judge", "hybrid")]


def lab(mode, jr):
    return f"{mode}:{jr}" if jr else mode


def score_and_cost(r, mode, jr):
    """(verifier score, judge cost incurred, judge latency incurred)"""
    if not r["heur_ok"]:
        return 0.0, 0.0, 0.0
    conf = r["cheap"]["conf"]
    if mode == "heuristic":
        return 1.0, 0.0, 0.0
    if mode == "self":
        return (conf or 0.0), 0.0, 0.0
    if mode == "hybrid" and conf is not None and conf >= GATE:
        return conf, 0.0, 0.0
    j = r["judges"][jr]
    return j["score"], j["cost"], j["latency_ms"]


def simulate(mode, jr, thr):
    cost = correct = esc = 0
    lats = []
    for r in rows:
        score, jc, jl = score_and_cost(r, mode, jr)
        e = score < thr
        cost += r["cheap"]["cost"] + jc + (r["strong"]["cost"] if e else 0)
        correct += r["strong"]["correct"] if e else r["cheap"]["correct"]
        lats.append(r["cheap"]["latency_ms"] + jl + (r["strong"]["latency_ms"] if e else 0))
        esc += e
    return cost, correct / N, esc / N, lats


cheap_cost = sum(r["cheap"]["cost"] for r in rows)
cheap_acc = sum(r["cheap"]["correct"] for r in rows) / N
strong_cost = sum(r["strong"]["cost"] for r in rows)
strong_acc = sum(r["strong"]["correct"] for r in rows) / N


def mk(combo, thr, cost, acc, esc, lats):
    return {"combo": combo, "thr": thr, "cost_usd": cost, "accuracy": acc, "escalation": esc,
            "cost_reduction_pct": (1 - cost / strong_cost) * 100,
            "quality_retained_pct": acc / strong_acc * 100,
            "p50_ms": pct(lats, 50), "p95_ms": pct(lats, 95)}


table = [
    mk("always-cheap", None, cheap_cost, cheap_acc, 0.0, [r["cheap"]["latency_ms"] for r in rows]),
    mk("always-strong", None, strong_cost, strong_acc, 1.0, [r["strong"]["latency_ms"] for r in rows]),
]
curves = {}
for mode, jr in COMBOS:
    curves[lab(mode, jr)] = []
    for thr in THRESHOLDS:
        cost, acc, esc, lats = simulate(mode, jr, thr)
        curves[lab(mode, jr)].append((cost, acc))
        table.append(mk(lab(mode, jr), thr, cost, acc, esc, lats))

df = pd.DataFrame(table)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
print(f"\nN={N} queries | hybrid gate={GATE} | judges compared: {JUDGES}")
print(f"cheap answers with NO parseable confidence: {sum(r['cheap']['conf'] is None for r in rows)}/{N}\n")
print(df.round(4).to_string(index=False))
df.to_csv("results/summary.csv", index=False)

print("\nBest operating point per verifier (max cost reduction with >=99% quality retained):")
for mode, jr in COMBOS:
    sub = df[(df.combo == lab(mode, jr)) & (df.quality_retained_pct >= 99)]
    if sub.empty:
        print(f"  {lab(mode, jr):22s} none reaches 99% quality")
    else:
        b = sub.sort_values("cost_reduction_pct", ascending=False).iloc[0]
        print(f"  {lab(mode, jr):22s} t={b.thr}: cost {b.cost_reduction_pct:+.1f}% | quality {b.quality_retained_pct:.1f}% | escalation {b.escalation:.0%}")

print("\nVerifier discrimination (mean score when cheap was RIGHT vs WRONG; bigger gap = better):")
def show(name, getter):
    right = [getter(r) for r in rows if r["cheap"]["correct"] and getter(r) is not None]
    wrong = [getter(r) for r in rows if not r["cheap"]["correct"] and getter(r) is not None]
    if right and wrong:
        print(f"  {name:16s} right={st.mean(right):.3f} (n={len(right)})  wrong={st.mean(wrong):.3f} (n={len(wrong)})")
show("self-confidence", lambda r: r["cheap"]["conf"])
for j in JUDGES:
    show(j, lambda r, j=j: r["judges"][j]["score"] if j in r["judges"] else None)
    avg = [r["judges"][j]["cost"] for r in rows if j in r["judges"]]
    print(f"  {'':16s} avg cost per judge call: ${st.mean(avg):.6f}")

plt.figure(figsize=(9, 6))
for name, pts in curves.items():
    plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, label=f"router[{name}]")
plt.scatter([cheap_cost], [cheap_acc], c="k", marker="s", label="always-cheap")
plt.scatter([strong_cost], [strong_acc], c="r", marker="*", s=140, label="always-strong")
plt.xlabel("Total cost (USD), judge included")
plt.ylabel("Accuracy")
plt.title(f"Cost vs accuracy ({N} MMLU questions)")
plt.legend(fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/cost_vs_accuracy.png", dpi=150)
print("\nSaved results/summary.csv and results/cost_vs_accuracy.png")
