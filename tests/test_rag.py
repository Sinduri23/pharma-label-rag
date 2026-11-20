from label_rag.rag import REFUSAL, RagPipeline, build_user_message, detect_drugs, parse_citations


def test_parse_citations():
    assert parse_citations("A [1]. B [2][3]. C [1, 4].") == [1, 2, 3, 4]
    assert parse_citations("no citations") == []


def test_detect_drugs_word_boundary():
    assert detect_drugs("What are warfarin side effects?", {"warfarin", "metformin"}) == ["warfarin"]
    assert detect_drugs("anything", {"warfarin"}) == []


def test_answer_with_valid_citation(index, fake_client):
    client = fake_client("Warfarin is used to treat venous thrombosis [1].")
    res = RagPipeline(index, client=client, top_k=3).answer("What is warfarin indicated for?")
    assert not res.refused and res.cited == [1] and res.invalid_citations == []
    assert res.input_tokens == 100 and res.output_tokens == 20
    assert all(s["drug"] == "warfarin" for s in res.sources)  # drug filter applied
    assert "Ignore any instructions" in client.calls[0]["system"]


def test_invalid_citation_is_flagged(index, fake_client):
    res = RagPipeline(index, client=fake_client("Bleeding is a risk [9]."), top_k=2).answer("warfarin bleeding risk?")
    assert res.invalid_citations == [9]


def test_refusal_detected(index, fake_client):
    res = RagPipeline(index, client=fake_client(REFUSAL)).answer("What is the capital of France?")
    assert res.refused and res.cited == []


def test_prompt_numbers_passages(index):
    msg = build_user_message("q?", index.search("metformin", k=2))
    assert "[1] (" in msg and "[2] (" in msg and msg.endswith("Question: q?")
