"""Pass 2: add ONE judge's verdicts to the cached cheap answers. Resumable; stops cleanly at the daily cap."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from app.llm import DailyLimitError  # noqa: E402
from app.verifier import judge  # noqa: E402
from run_benchmark import build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, help="judge | judge_120b | judge_20b")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open("results/raw.jsonl")]
    out = Path(f"results/judges_{args.role}.jsonl")
    done = {json.loads(line)["idx"] for line in out.open()} if out.exists() else set()
    todo = [r for r in rows if r["heur_ok"] and args.role not in r.get("judges", {})
            and r["idx"] not in done and "ans" in r["cheap"]]
    print(f"{args.role}: {len(done)} saved, {len(todo)} to judge")
    if not todo:
        return

    ds = load_dataset("cais/mmlu", "all", split="test")

    def work(r):
        score, res = judge(build_prompt(ds[r["idx"]]), r["cheap"]["ans"], args.role)
        return {"idx": r["idx"], "role": args.role,
                "res": {"score": score, "cost": res.cost_usd, "latency_ms": res.latency_ms,
                        "in": res.input_tokens, "out": res.output_tokens}}

    failed = 0
    with out.open("a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, r): r["idx"] for r in todo}
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
            if n % 20 == 0:
                print(f"{n}/{len(todo)} (failed so far: {failed})")
    print(f"finished. failed={failed}. Re-run the same command to retry failures.")


if __name__ == "__main__":
    main()
