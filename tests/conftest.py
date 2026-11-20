from types import SimpleNamespace

import pytest

from label_rag.chunking import chunk_label
from label_rag.embeddings import HashingEmbedder
from label_rag.index import VectorIndex

LABELS = [
    {"set_id": "a1", "drug": "metformin", "sections": {
        "indications_and_usage": "Metformin is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
        "contraindications": "Metformin is contraindicated in patients with severe renal impairment and metabolic acidosis."}},
    {"set_id": "b2", "drug": "warfarin", "sections": {
        "indications_and_usage": "Warfarin is indicated for prophylaxis and treatment of venous thrombosis and pulmonary embolism.",
        "adverse_reactions": "The most serious risk of warfarin is bleeding. Hemorrhage can occur at any site."}},
    {"set_id": "c3", "drug": "sertraline", "sections": {
        "indications_and_usage": "Sertraline is indicated for major depressive disorder, obsessive compulsive disorder and panic disorder."}},
]


@pytest.fixture
def index():
    chunks = [c for lab in LABELS for c in chunk_label(lab)]
    return VectorIndex.build(HashingEmbedder(), chunks)


class FakeClient:
    """Minimal stand-in for anthropic.Anthropic().messages.create."""

    def __init__(self, text):
        self.text = text
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


@pytest.fixture
def fake_client():
    return FakeClient
