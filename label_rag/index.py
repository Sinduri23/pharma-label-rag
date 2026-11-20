from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorIndex:
    """FAISS inner-product index over L2-normalised embeddings (= cosine similarity)."""

    def __init__(self, embedder, chunks: list[Chunk], index: faiss.Index):
        self.embedder = embedder
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, embedder, chunks: list[Chunk], batch_size: int = 256) -> "VectorIndex":
        vecs = np.vstack(
            [embedder.encode([c.embed_text for c in chunks[i : i + batch_size]]) for i in range(0, len(chunks), batch_size)]
        )
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(embedder, chunks, index)

    @property
    def drugs(self) -> set[str]:
        return {c.drug for c in self.chunks}

    def search(self, query: str, k: int = 5, drugs: list[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks. If `drugs` is given, restrict to chunks of those drugs (exact, flat search)."""
        q = self.embedder.encode([query])
        fetch = len(self.chunks) if drugs else min(k, len(self.chunks))
        scores, ids = self.index.search(q, fetch)
        out: list[tuple[Chunk, float]] = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            chunk = self.chunks[i]
            if drugs and chunk.drug not in drugs:
                continue
            out.append((chunk, float(score)))
            if len(out) == k:
                break
        return out

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))

    @classmethod
    def load(cls, directory: str | Path, embedder) -> "VectorIndex":
        d = Path(directory)
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        return cls(embedder, chunks, faiss.read_index(str(d / "faiss.index")))
