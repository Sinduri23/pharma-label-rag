"""Chunk the downloaded labels, embed them and write a FAISS index."""
from __future__ import annotations

import argparse
import json

from . import config
from .chunking import chunk_label
from .embeddings import get_embedder
from .index import VectorIndex


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(config.RAW_LABELS_PATH))
    p.add_argument("--out", default=str(config.INDEX_DIR))
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=1, help="overlap in sentences")
    args = p.parse_args()

    labels = json.load(open(args.labels))
    chunks = [c for lab in labels for c in chunk_label(lab, args.max_chars, args.overlap)]
    print(f"{len(labels)} labels -> {len(chunks)} chunks")
    VectorIndex.build(get_embedder(), chunks).save(args.out)
    print(f"index saved to {args.out}")


if __name__ == "__main__":
    main()
