"""Run the full evaluation and save per-question results for error analysis.

Example:
    python -m label_rag.run_eval --judge --out eval/results.json
    python -m label_rag.run_eval --no-drug-filter --retrieval-only     # ablation
"""
from __future__ import annotations

import argparse
import json

from . import config
from .embeddings import get_embedder
from .evaluation import eval_generation, eval_retrieval, load_jsonl
from .index import VectorIndex
from .rag import RagPipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", nargs="+", default=["eval/gold_questions.jsonl", "eval/unanswerable.jsonl"])
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--judge", action="store_true", help="LLM-judge groundedness (extra API calls)")
    p.add_argument("--no-drug-filter", action="store_true")
    p.add_argument("--retrieval-only", action="store_true", help="skip generation (no API calls)")
    p.add_argument("--include-unreviewed", action="store_true")
    p.add_argument("--out", default="eval/results.json")
    args = p.parse_args()

    rows = [r for path in args.gold for r in load_jsonl(path, only_reviewed=not args.include_unreviewed)]
    index = VectorIndex.load(config.INDEX_DIR, get_embedder())
    results = {"config": vars(args), "retrieval": eval_retrieval(index, rows, args.k, not args.no_drug_filter)}
    r = results["retrieval"]
    print(f"retrieval  n={r['n']}  hit@{args.k}={r['hit_at_k']:.3f}  mrr={r['mrr']:.3f}")

    if not args.retrieval_only:
        pipe = RagPipeline(index, top_k=args.k)
        g = eval_generation(pipe, rows, k=args.k, judge=args.judge)
        results["generation"] = g
        for key in ["false_refusal_rate", "correct_refusal_rate", "citation_valid_rate", "groundedness_rate",
                    "hallucination_rate", "avg_latency_ms", "avg_tokens"]:
            print(f"{key:22s} {g[key]}")

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}  (open it and categorise the failures: that is your error analysis)")


if __name__ == "__main__":
    main()
