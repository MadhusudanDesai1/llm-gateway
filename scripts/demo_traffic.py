"""Send a mix of prompts to a running gateway so the dashboard has data to show."""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

EASY = [
    "What is the capital of Australia?",
    "Convert 72 F to Celsius.",
    "Who wrote Pride and Prejudice?",
    "What year did the Berlin Wall fall?",
    "What is the chemical symbol for tungsten?",
    "How many bits are in a kilobyte?",
    "What causes the seasons on Earth?",
    "Is 391 a prime number?",
    "Name three sorting algorithms with O(n log n) average time.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
]
HARD = [
    "Explain why the Banach-Tarski paradox does not contradict measure theory.",
    "In the Linux kernel, what is the difference between a spinlock and a mutex, and when is each used?",
    "Find all local maxima of f(x) = x^2 * e^(-x) for x >= 0.",
    "What is the best known asymptotic time complexity for matrix multiplication, and who improved it most recently?",
    "Why does the CAP theorem force a trade-off between consistency and availability during a partition?",
    "Explain how the Raft consensus algorithm handles a leader failure.",
]
FORCED = [
    "What is 12*12?",
    "Name the largest planet in the solar system.",
    "What is the boiling point of water in Kelvin?",
    "Who painted the Mona Lisa?",
]


def post(prompt, threshold=None):
    body = {"prompt": prompt}
    if threshold is not None:
        body["threshold"] = threshold
    req = urllib.request.Request(f"{URL}/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


for label, prompts, thr in [("easy", EASY, None), ("hard", HARD, None), ("forced-strong", FORCED, 1.5)]:
    for p in prompts:
        r = post(p, thr)
        print(f"{label:13s} {r['route']:10s} score={r['verifier_score']:.2f} {r['latency_ms']:>5}ms "
              f"${r['cost_usd']:.6f} degraded={r['degraded']} | {r['answer'][:45].replace(chr(10), ' ')!r}")
