# Pharma Label RAG Assistant

Retrieval-augmented question answering over **public FDA drug labels** (openFDA), with
section-level source citations, refusal on out-of-scope questions, an automated evaluation harness,
a FastAPI service, CI and Docker.

> Informational prototype only. Not medical advice. Uses public, non-sensitive data only.

## Architecture

```mermaid
flowchart LR
  A[openFDA labels] --> B[Section-aware chunking]
  B --> C[Embeddings]
  C --> D[(FAISS index)]
  Q[Question] --> E[Drug detection + retrieval top-k]
  D --> E
  E --> F[Claude: answer using ONLY numbered passages, cite [n]]
  F --> G[Citation validation + refusal detection]
  G --> H[FastAPI /ask + logs: latency, tokens]
```

Design choices (and why):
- **Chunk by label section** (indications, contraindications, ...) so answers cite a meaningful source and sections never mix.
- **Embed `drug + section + text`**, so queries like "warfarin bleeding" match the right label.
- **Drug-name filter**: if the question names a drug in the corpus, retrieval is restricted to that drug (ablation: `--no-drug-filter`).
- **Grounded prompt**: numbered passages, mandatory citations, exact refusal string, passages treated as untrusted (prompt-injection defence).
- **Programmatic checks**: every `[n]` must refer to a retrieved passage; refusals are detected exactly.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-embeddings.txt
export ANTHROPIC_API_KEY=...            # never commit this

python -m label_rag.ingest              # download labels from openFDA -> data/labels.json
python -m label_rag.build_index         # chunk + embed + FAISS -> data/index/
uvicorn label_rag.api:app --reload      # http://localhost:8000/docs

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What are the contraindications of metformin?"}'
```

Offline smoke test without downloading a model: `EMBED_BACKEND=hashing` (do not report eval numbers from it).

## Evaluation

```bash
python -m label_rag.make_questions --n 80      # drafts questions -> eval/gold_questions.jsonl (reviewed=false)
# >>> open the file, fix/delete weak questions, set "reviewed": true on the ones you trust <<<
python -m label_rag.run_eval --judge           # retrieval + generation + LLM-judged groundedness
python -m label_rag.run_eval --retrieval-only --no-drug-filter   # ablation, no API calls
```

Metrics: retrieval **hit@k / MRR**, **citation validity**, **false-refusal** and **correct-refusal** rates
(`eval/unanswerable.jsonl`), **groundedness / hallucination rate** (LLM judge), latency and tokens.
Per-question results are saved to `eval/results.json` for error analysis.

### Results

> TODO: fill in from YOUR runs. Do not copy numbers you did not measure.

| Variant | hit@5 | MRR | Groundedness | Citation valid | Correct refusal |
|---|---|---|---|---|---|
| Baseline (no drug filter, 1200-char chunks) | | | | | |
| + drug filter | | | | | |
| + [your next change] | | | | | |

### Error analysis

> TODO: categorise failures from `eval/results.json` (e.g. wrong section retrieved, multi-chunk answer, over-refusal, unsupported claim) and note what you changed.

## Tests and CI

```bash
pytest -q
```
GitHub Actions runs the test suite on every push (`.github/workflows/ci.yml`). Tests use a fake LLM client and an offline embedder, so no API key is needed.

## Docker

```bash
docker build -t label-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -v $(pwd)/data:/app/data label-rag
```

## Responsible AI and limitations
- Public data only; no patient or proprietary data. The assistant must not give personal medical advice.
- Answers are limited to the retrieved label text; one label version per drug, which may be outdated.
- LLM-as-judge groundedness is itself imperfect; spot-check judged rows by hand.
- Retrieval uses a single top-k; multi-section questions can be missed.
