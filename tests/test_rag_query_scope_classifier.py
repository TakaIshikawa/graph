from graph.rag.query_scope_classifier import classify_query_scope


def test_query_scope_classifier_detects_comparative_queries():
    report = classify_query_scope("Compare Postgres versus SQLite for mobile sync. Which is better?")

    assert report["primary_scope"] == "comparative"
    assert report["matched_cues"] == ["versus", "compare", "better"]
    assert report["confidence"] == 0.85


def test_query_scope_classifier_detects_procedural_queries():
    report = classify_query_scope("How to implement a retry workflow?")

    assert report["primary_scope"] == "procedural"
    assert report["matched_cues"] == ["how to", "workflow", "implement"]


def test_query_scope_classifier_defaults_to_narrow():
    assert classify_query_scope("Redis maxmemory policy") == {
        "primary_scope": "narrow",
        "matched_cues": [],
        "confidence": 0.55,
        "recommended_retrieval_strategy": "retrieve focused exact-match evidence",
    }
