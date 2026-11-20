from __future__ import annotations

import json
import logging
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from . import config
from .embeddings import get_embedder
from .index import VectorIndex
from .rag import RagPipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("label_rag")

app = FastAPI(title="Pharma Label RAG", description="Cited Q&A over public FDA drug labels (informational only).")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline(VectorIndex.load(config.INDEX_DIR, get_embedder()))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    result = pipeline.answer(req.question, k=req.k)
    log.info(
        json.dumps(
            {
                "event": "ask",
                "latency_ms": round(result.latency_ms),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "refused": result.refused,
                "invalid_citations": result.invalid_citations,
            }
        )
    )
    return result.to_dict()
