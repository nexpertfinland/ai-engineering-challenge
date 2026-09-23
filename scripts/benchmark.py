"""Compare the original and improved solutions without changing the scorer.

uv run python scripts/benchmark.py --split retrieval
uv run python scripts/benchmark.py --split dev --pipeline both
uv run python scripts/benchmark.py --split hidden --pipeline both
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.baseline import BaselinePipeline
from rag.evaluation import evaluate_dev, retrieval_report
from rag.llm import DEFAULT_MODEL, PROVIDER, ensure_model_ready
from rag.pipeline import RAGPipeline


def load_data():
    paths = [ROOT / "data/corpus.json", ROOT / "data/dev_qa.json", ROOT / "eval/test_qa.json"]
    if not all(p.exists() for p in paths):
        from scripts.prepare_data import build
        build(20, 40, 80, 42)
    # Held-out labels are only read by the unchanged official harness.
    return tuple(json.loads(p.read_text(encoding="utf-8")) for p in paths[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["retrieval", "dev", "hidden"], default="retrieval")
    parser.add_argument("--pipeline", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus, dev = load_data()
    if args.split != "retrieval":
        ensure_model_ready()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "commit": commit, "python": platform.python_version(),
        "provider": PROVIDER if args.split != "retrieval" else None,
        "model": DEFAULT_MODEL if args.split != "retrieval" else None,
        "corpus_sha256": hashlib.sha256((ROOT / "data/corpus.json").read_bytes()).hexdigest(),
        "scorer_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "eval").glob("*.py"))},
        "results": {},
    }
    output = args.output or ROOT / "results" / f"{args.split}-{args.pipeline}.json"
    for name, factory in [("baseline", BaselinePipeline), ("improved", RAGPipeline)]:
        if args.pipeline not in (name, "both"):
            continue
        pipeline = factory(corpus)
        print(f"\n{name} / {args.split}", flush=True)
        if args.split == "retrieval":
            result = {f"top_{k}": retrieval_report(pipeline.retrieve, dev, k) for k in (1, 3)}
        elif args.split == "dev":
            result = evaluate_dev(pipeline.answer_question, dev)
        else:
            from eval.harness import run_hidden_eval
            result = run_hidden_eval(pipeline.answer_question)
        report["results"][name] = result
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2), flush=True)
        if result.get("errors", 0):
            raise RuntimeError(f"Evaluation had {result['errors']} pipeline errors; inspect {output}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
