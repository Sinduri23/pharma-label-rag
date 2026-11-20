"""Evaluation: retrieval quality, citation validity, refusal behaviour and LLM-judged groundedness."""
from __future__ import annotations

import json
import re
from pathlib import Path

JUDGE_PROMPT = """You are a strict grader. Given context passages and an answer, decide whether EVERY factual
claim in the answer is supported by the passages.
Reply with JSON only: {"supported": true|false, "unsupported_claims": ["..."]}"""


def load_jsonl(path: str | Path, only_reviewed: bool = True) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if only_reviewed:
        rows = [r for r in rows if r.get("reviewed", False)]
    return rows


def eval_retrieval(index, rows: list[dict], k: int = 5, use_drug_filter: bool = True) -> dict:
    """Hit@k and MRR: is a chunk from the expected drug AND section in the top-k?"""
    from .rag import detect_drugs

    answerable = [r for r in rows if r.get("answerable", True)]
    per_row, hits, rr_sum = [], 0, 0.0
    for r in answerable:
        drugs = detect_drugs(r["question"], index.drugs) if use_drug_filter else None
        retrieved = index.search(r["question"], k, drugs or None)
        rank = next(
            (i for i, (c, _) in enumerate(retrieved, 1) if c.drug == r["expected_drug"] and c.section == r["expected_section"]),
            None,
        )
        hits += rank is not None
        rr_sum += 1 / rank if rank else 0.0
        per_row.append({"question": r["question"], "expected": [r["expected_drug"], r["expected_section"]], "rank": rank})
    n = max(len(answerable), 1)
    return {"k": k, "drug_filter": use_drug_filter, "n": len(answerable), "hit_at_k": hits / n, "mrr": rr_sum / n, "rows": per_row}


def judge_groundedness(client, model: str, question: str, sources: list[dict], answer: str) -> dict:
    passages = "\n\n".join(f"[{s['n']}] {s['text']}" for s in sources)
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer: {answer}"}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"supported": None, "unsupported_claims": []}
    except json.JSONDecodeError:
        return {"supported": None, "unsupported_claims": []}


def eval_generation(pipeline, rows: list[dict], k: int = 5, judge: bool = False, judge_model: str | None = None) -> dict:
    per_row = []
    for r in rows:
        res = pipeline.answer(r["question"], k=k)
        entry = {
            "question": r["question"],
            "answerable": r.get("answerable", True),
            "answer": res.answer,
            "refused": res.refused,
            "cited": res.cited,
            "invalid_citations": res.invalid_citations,
            "latency_ms": round(res.latency_ms),
            "tokens": res.input_tokens + res.output_tokens,
        }
        if entry["answerable"] and not res.refused:
            entry["citation_ok"] = bool(res.cited) and not res.invalid_citations
            if judge:
                verdict = judge_groundedness(pipeline.client, judge_model or pipeline.model, r["question"], res.sources, res.answer)
                entry["grounded"] = verdict.get("supported")
                entry["unsupported_claims"] = verdict.get("unsupported_claims", [])
        per_row.append(entry)

    ans = [e for e in per_row if e["answerable"]]
    unans = [e for e in per_row if not e["answerable"]]
    answered = [e for e in ans if not e["refused"]]
    graded = [e for e in answered if e.get("grounded") is not None]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "false_refusal_rate": sum(e["refused"] for e in ans) / max(len(ans), 1),
        "correct_refusal_rate": sum(e["refused"] for e in unans) / max(len(unans), 1),
        "citation_valid_rate": sum(e.get("citation_ok", False) for e in answered) / max(len(answered), 1),
        "groundedness_rate": (sum(bool(e["grounded"]) for e in graded) / len(graded)) if graded else None,
        "hallucination_rate": (1 - sum(bool(e["grounded"]) for e in graded) / len(graded)) if graded else None,
        "avg_latency_ms": sum(e["latency_ms"] for e in per_row) / max(len(per_row), 1),
        "avg_tokens": sum(e["tokens"] for e in per_row) / max(len(per_row), 1),
        "rows": per_row,
    }
