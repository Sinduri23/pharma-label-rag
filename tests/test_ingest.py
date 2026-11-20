from label_rag.ingest import normalize


def test_normalize_joins_lists_and_skips_missing():
    raw = {"set_id": "s1", "openfda": {"generic_name": ["METFORMIN HYDROCHLORIDE"], "brand_name": ["Glucophage"]},
           "indications_and_usage": ["Part one.", "Part two."], "warnings": []}
    out = normalize(raw, "Metformin")
    assert out["drug"] == "metformin" and out["brand"] == "Glucophage"
    assert out["sections"] == {"indications_and_usage": "Part one. Part two."}
