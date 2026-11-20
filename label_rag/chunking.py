"""Section-aware chunking: never mixes text from two label sections."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    drug: str
    section: str
    text: str
    set_id: str = ""

    @property
    def embed_text(self) -> str:
        """Text that gets embedded: drug + section name give the embedding useful context."""
        return f"{self.drug} {self.section.replace('_', ' ')}: {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_text(text: str, max_chars: int = 1200, overlap_sentences: int = 1) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    for s in _SENT.split(text):
        if len(s) > max_chars:  # pathological run-on sentence: hard split
            sentences.extend(s[i : i + max_chars] for i in range(0, len(s), max_chars))
        else:
            sentences.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(x) + 1 for x in cur)
            if cur_len + len(s) + 1 > max_chars:
                cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_label(label: dict, max_chars: int = 1200, overlap_sentences: int = 1) -> list[Chunk]:
    out = []
    for section, text in label["sections"].items():
        for i, piece in enumerate(split_text(text, max_chars, overlap_sentences)):
            out.append(
                Chunk(
                    chunk_id=f"{_slug(label['drug'])}:{section}:{i}",
                    drug=label["drug"],
                    section=section,
                    text=piece,
                    set_id=label.get("set_id", ""),
                )
            )
    return out
