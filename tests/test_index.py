from label_rag.embeddings import HashingEmbedder
from label_rag.index import VectorIndex


def test_retrieves_right_section(index):
    hits = index.search("what is warfarin indicated for", k=1)
    assert hits[0][0].drug == "warfarin"
    assert hits[0][0].section == "indications_and_usage"


def test_drug_filter_restricts_results(index):
    hits = index.search("indicated for", k=3, drugs=["sertraline"])
    assert hits and all(c.drug == "sertraline" for c, _ in hits)


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, HashingEmbedder())
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    assert loaded.search("bleeding risk", k=1)[0][0].drug == "warfarin"
