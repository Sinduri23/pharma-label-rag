from label_rag.evaluation import eval_generation, eval_retrieval, judge_groundedness
from label_rag.rag import REFUSAL, RagPipeline

ROWS = [
    {"question": "What is metformin indicated for?", "expected_drug": "metformin", "expected_section": "indications_and_usage", "answerable": True},
    {"question": "What is the bleeding risk of warfarin?", "expected_drug": "warfarin", "expected_section": "adverse_reactions", "answerable": True},
    {"question": "What is the capital of France?", "answerable": False},
]


def test_retrieval_metrics(index):
    r = eval_retrieval(index, ROWS, k=3)
    assert r["n"] == 2 and r["hit_at_k"] == 1.0 and r["mrr"] > 0.5


def test_generation_metrics_unanswerable_refused(index, fake_client):
    pipe = RagPipeline(index, client=fake_client(REFUSAL))
    g = eval_generation(pipe, ROWS)
    assert g["correct_refusal_rate"] == 1.0
    assert g["false_refusal_rate"] == 1.0  # fake always refuses, so answerable ones are false refusals


def test_judge_parses_json(index, fake_client):
    client = fake_client('Sure: {"supported": false, "unsupported_claims": ["x"]}')
    v = judge_groundedness(client, "m", "q", [{"n": 1, "text": "t"}], "a")
    assert v["supported"] is False and v["unsupported_claims"] == ["x"]
    assert judge_groundedness(fake_client("garbage"), "m", "q", [], "a")["supported"] is None
