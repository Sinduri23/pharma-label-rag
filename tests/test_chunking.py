from label_rag.chunking import chunk_label, split_text


def test_chunks_respect_max_chars_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, max_chars=300, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # last sentence of a chunk is repeated at the start of the next
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_runon_sentence_is_hard_split():
    chunks = split_text("x" * 2500, max_chars=1000)
    assert all(len(c) <= 1000 for c in chunks) and len(chunks) == 3


def test_empty_text():
    assert split_text("   ") == []


def test_chunk_label_metadata_and_unique_ids():
    label = {"set_id": "z", "drug": "Drug A", "sections": {"warnings": "One. " * 600, "contraindications": "None."}}
    chunks = chunk_label(label, max_chars=500)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.section for c in chunks} == {"warnings", "contraindications"}
    assert all(c.drug == "Drug A" and c.set_id == "z" for c in chunks)
    assert "contraindications" in chunks[-1].embed_text
