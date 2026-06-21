"""Full 456-question TreatmentPC reproduction test for ATHENA-R1.

Runs the release model with native self-extraction (Backend.ATHENA) at
T=0.7, presence_penalty=0. GPT-mapped scoring is skipped here (needs Azure).

Per-question progress is appended to PROGRESS_FILE so the run is resumable.

Usage:
    PYTHONPATH=src python -u tests/test_reproducibility_full.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VLLM_URL = os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1")
TOOL_SERVER = os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080")
# DATASET_PATH must be set explicitly — the personalized-treatment MCQ file is
# not redistributed with this repo (see README for sourcing instructions).
DATASET_PATH = os.environ.get("DATASET_PATH", "")
PROGRESS_FILE = Path(
    os.environ.get(
        "PROGRESS_FILE",
        str(Path(__file__).parent / "reproducibility_full.jsonl"),
    )
)
CONCURRENT_QS = int(os.environ.get("CONCURRENT_QS", "4"))
MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS", "9999"))
# Per-Q wall-clock budget (in seconds). 0 disables the cap. Defaults to 12 min,
# which is well above typical per-question latency but caps the long-tail
# stuck-question failure mode.
PER_QUESTION_TIMEOUT = float(os.environ.get("PER_QUESTION_TIMEOUT", "720"))


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}
    done = {}
    for line in progress_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        done[r["id"]] = r
    return done


def get_model_id() -> str:
    r = requests.get(f"{VLLM_URL.rstrip('/v1')}/v1/models", timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


# How many times to re-attempt a question that produced no answer (empty
# prediction or a wall-clock timeout). Under K-way concurrency vLLM very
# occasionally leaves one request stuck until the per-question timeout fires;
# the same question almost always succeeds on a fresh attempt (different
# concurrency timing), so a single retry keeps a rare infra hiccup from
# costing a real prediction. Set 0 to disable.
STUCK_Q_RETRIES = int(os.environ.get("STUCK_Q_RETRIES", "2"))


def _attempt_one(agent, question, options, gold):
    """One Stage-1 + Stage-2 attempt. Returns (pred, correct, rounds, stuck)
    where ``stuck`` is True iff the run produced no usable answer (empty /
    wall-clock cancelled) and is worth retrying."""
    timeout = PER_QUESTION_TIMEOUT if PER_QUESTION_TIMEOUT > 0 else None
    result = agent.answer(question, temperature=0.7, max_round=40, timeout=timeout)
    if not result.answer:
        return "", False, result.rounds_used, True
    from athena_r1 import Backend

    pred = agent.map_to_option(result.conversation, options, backend=Backend.ATHENA)
    return pred, pred == gold, result.rounds_used, bool(result.cancelled)


def evaluate_one(agent, idx: int, sample: dict, log_lock) -> dict:
    question = sample["question"]
    options = sample["options"]
    gold = sample["correct_answer"]

    t0 = time.time()
    pred, correct, rounds, retries = "", False, 0, 0
    for attempt in range(STUCK_Q_RETRIES + 1):
        retries = attempt  # number of re-attempts performed before this one
        try:
            pred, correct, rounds, stuck = _attempt_one(agent, question, options, gold)
            if not stuck:
                break  # got a real answer — done
            if attempt < STUCK_Q_RETRIES:
                with log_lock:
                    print(
                        f"  [{idx}] stuck (empty/timeout), retry {attempt + 1}/{STUCK_Q_RETRIES}",
                        flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            pred, correct, rounds = "", False, 0
            with log_lock:
                print(f"  [{idx}] EXCEPTION: {e!r}", flush=True)
            break

    elapsed = time.time() - t0
    rec = {
        "id": idx,
        "predict": pred,
        "label": gold,
        "correct": correct,
        "rounds": rounds,
        "elapsed_s": round(elapsed, 1),
    }
    if retries:
        rec["retries"] = retries
    return rec


def main() -> int:
    import threading

    from athena_r1 import AthenaR1

    data = read_jsonl(DATASET_PATH)[:MAX_QUESTIONS]
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(f"Total Qs: {len(data)}", flush=True)

    done = load_done(PROGRESS_FILE)
    print(f"Already done (resume): {len(done)}", flush=True)

    pending = [(i, sample) for i, sample in enumerate(data) if i not in done]
    print(f"Pending: {len(pending)}", flush=True)

    if not pending:
        print("All done — computing final stats from progress file ...", flush=True)
        return summarize(done)

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress_fh = PROGRESS_FILE.open("a", buffering=1)
    log_lock = threading.Lock()

    agent = AthenaR1(
        model=get_model_id(),
        vllm_url=VLLM_URL,
        tool_server=TOOL_SERVER,
        max_round=40,
    )
    agent.init()
    print(f"Agent ready. Running {len(pending)} Qs with concurrency={CONCURRENT_QS}", flush=True)

    with ThreadPoolExecutor(max_workers=CONCURRENT_QS) as ex:
        futures = {ex.submit(evaluate_one, agent, idx, s, log_lock): idx for idx, s in pending}
        for fut in as_completed(futures):
            rec = fut.result()
            done[rec["id"]] = rec
            with log_lock:
                progress_fh.write(json.dumps(rec) + "\n")
                progress_fh.flush()
                n_done_so_far = len(done)
                # Use only the cells we have computed (not resumed) for live acc:
                vals = list(done.values())
                n_corr = sum(1 for r in vals if r.get("correct"))
                acc = n_corr / len(vals)
                mark = "✓" if rec["correct"] else "✗"
                print(
                    f"  [{n_done_so_far}/{len(data)}] {mark} pred={rec['predict']!r:5} "
                    f"gold={rec['label']!r:3} acc={acc * 100:5.1f}% "
                    f"rounds={rec['rounds']} t={rec['elapsed_s']}s",
                    flush=True,
                )

    progress_fh.close()
    return summarize(done)


def summarize(done: dict) -> int:
    total = len(done)
    if total == 0:
        print("no results")
        return 1
    n_corr = sum(1 for r in done.values() if r.get("correct"))
    n_empty = sum(1 for r in done.values() if r.get("predict", "") == "")
    n_nonempty = total - n_empty
    acc_all = n_corr / total
    acc_ne = n_corr / max(1, n_nonempty)
    print()
    print("=" * 60)
    print(f"FINAL native accuracy: {n_corr}/{total} = {acc_all * 100:.2f}% (raw)")
    print(f"                       {n_corr}/{n_nonempty} = {acc_ne * 100:.2f}% (excl. empty)")
    print(f"  empty predictions:   {n_empty}")
    print("  reference native self-extract (pp=0): ~75.7%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
