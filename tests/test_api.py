from fastapi.testclient import TestClient

from label_rag.api import app, get_pipeline
from label_rag.rag import RagPipeline


def test_ask_endpoint(index, fake_client):
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(index, client=fake_client("Metformin treats type 2 diabetes [1]."))
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ask", json={"question": "What is metformin used for?"})
    assert r.status_code == 200 and r.json()["cited"] == [1]
    assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    app.dependency_overrides.clear()
