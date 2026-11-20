import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", DATA_DIR / "index"))
RAW_LABELS_PATH = DATA_DIR / "labels.json"

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "sbert")  # sbert | hashing (hashing = offline/tests only)
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
