"""Draft a synthetic question set from random chunks, for YOU to review and edit.

Every row is written with reviewed=false. Open the file, fix or delete bad questions,
then set reviewed=true on the ones you trust. run_eval only uses reviewed rows by default.
"""
from __future__ import annotations

import argparse
import json
import random

from . import config
from .index import VectorIndex
from .embeddings import get_embedder

PROMPT = """Write ONE specific question that can be answered using ONLY the passage below.
Rules: mention the drug name explicitly; the question must be self-contained; do not copy sentences verbatim.
Return just the question."""

SECTIONS = ["indications_and_usage", "contraindications", "dosage_and_administration", "adverse_reactions",
            "drug_interactions", "warnings_and_cautions", "overdosage"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", default="eval/gold_questions.jsonl")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    import anthropic

    client = anthropic.Anthropic()
    index = VectorIndex.load(config.INDEX_DIR, get_embedder())
    pool = [c for c in index.chunks if c.section in SECTIONS and len(c.text) > 300]
    random.Random(args.seed).shuffle(pool)

    with open(args.out, "a") as f:
        for c in pool[: args.n]:
            resp = client.messages.create(
                model=config.MODEL, max_tokens=120, system=PROMPT,
                messages=[{"role": "user", "content": f"Drug: {c.drug}\nPassage:\n{c.text}"}],
            )
            q = resp.content[0].text.strip()
            row = {"question": q, "expected_drug": c.drug, "expected_section": c.section,
                   "answerable": True, "source_chunk": c.chunk_id, "reviewed": False}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.n} DRAFT questions to {args.out}. Review them, then set reviewed=true.")


if __name__ == "__main__":
    main()
