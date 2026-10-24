"""Pass 1: run N MMLU questions through cheap and strong; cache results.
Judges run separately (run_judges.py) so each model's daily token budget is used independently."""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from app.llm import DailyLimitError  # noqa: E402
from app.router import run_cheap, run_strong  # noqa: E402
from app.verifier import heuristics  # noqa: E402

OUT = Path("results/raw.jsonl")
LETTERS = "ABCD"


def build_prompt(row):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return f"{row['question']}\n\n{opts}\n\nReply with 'Answer: <letter>'."


def extract(text):
    m = re.search(r"Answer:\s*\(?([A-D])\b", text, re.I) or re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else None


def process(idx, row):
    prompt = build_prompt(row)
    gold = LETTERS[row["answer"]]

    c_res, c_ans, c_conf = run_cheap(prompt)
    heur_ok, _ = heuristics(c_ans, c_res.finish_reason)
    s_res = run_strong(prompt)

    c_pred, s_pred = extract(c_ans), extract(s_res.text)
    return {
        "idx": idx,
        "subject": row["subject"],
        "gold": gold,
        "heur_ok": heur_ok,
        "cheap": {"pred": c_pred, "correct": c_pred == gold, "conf": c_conf, "cost": c_res.cost_usd,
                  "latency_ms": c_res.latency_ms, "in": c_res.input_tokens, "out": c_res.output_tokens,
                  "ans": c_ans, "tail": c_res.text[-80:]},
        "judges": {},
        "strong": {"pred": s_pred, "correct": s_pred == gold, "cost": s_res.cost_usd,
                   "latency_ms": s_res.latency_ms, "in": s_res.input_tokens, "out": s_res.output_tokens},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = random.Random(args.seed).sample(range(len(ds)), args.n)

    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["idx"] for line in OUT.open()}
    todo = [i for i in idxs if i not in done]
    print(f"{len(done)} already done, {len(todo)} to run")

    failed = 0
    with OUT.open("a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process, i, ds[i]): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except DailyLimitError as e:
                print(f"STOPPING: daily token limit reached -> {e}\nProgress is saved; re-run the same command later.")
                ex.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                failed += 1
                print(f"idx {futs[fut]} failed: {type(e).__name__}: {str(e)[:200]}")
            if n % 10 == 0:
                print(f"{n}/{len(todo)} (failed so far: {failed})")
    print(f"finished. failed={failed}. Re-run the same command to retry failures.")


if __name__ == "__main__":
    main()
