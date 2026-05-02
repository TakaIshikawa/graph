from __future__ import annotations

from graph.rag import classify_query_intent


def test_classify_query_intent_detects_lookup_with_full_text_mode():
    result = classify_query_intent("Find the exact note about vector clocks")

    assert result["intent"] == "lookup"
    assert result["confidence"] == 0.95
    assert result["matched_cues"] == ["find", "exact"]
    assert result["suggested_search_mode"] == "full_text"
    assert result["suggested_metadata_filters"] == {}


def test_classify_query_intent_detects_comparison_with_hybrid_mode():
    result = classify_query_intent("Compare CRDTs versus operational transforms")

    assert result["intent"] == "comparison"
    assert result["matched_cues"] == ["compare", "versus"]
    assert result["suggested_search_mode"] == "hybrid"


def test_classify_query_intent_detects_timeline_and_date_filters():
    result = classify_query_intent("Timeline of graph database notes after 2021 before 2024")

    assert result["intent"] == "timeline"
    assert result["matched_cues"] == ["timeline", "before", "year"]
    assert result["suggested_search_mode"] == "hybrid"
    assert result["suggested_metadata_filters"] == {
        "date": {"years": ["2021", "2024"], "before": "2024", "after": "2021"}
    }


def test_classify_query_intent_detects_how_to_with_semantic_mode():
    result = classify_query_intent("How to build a reading workflow from saved papers")

    assert result["intent"] == "how_to"
    assert result["matched_cues"] == ["how to", "guide"]
    assert result["suggested_search_mode"] == "semantic"


def test_classify_query_intent_detects_definition_with_semantic_mode():
    result = classify_query_intent("What is source diversity reranking?")

    assert result["intent"] == "definition"
    assert result["matched_cues"] == ["what is"]
    assert result["suggested_search_mode"] == "semantic"


def test_classify_query_intent_detects_contradiction_check_with_hybrid_mode():
    result = classify_query_intent("Which findings contradict or conflict with the benchmark?")

    assert result["intent"] == "contradiction_check"
    assert result["matched_cues"] == ["contradicts", "conflicts"]
    assert result["suggested_search_mode"] == "hybrid"


def test_classify_query_intent_detects_exploratory_queries():
    result = classify_query_intent("Explore related themes around graph embeddings")

    assert result["intent"] == "exploratory"
    assert result["matched_cues"] == ["explore", "related", "themes"]
    assert result["suggested_search_mode"] == "semantic"


def test_classify_query_intent_extracts_tag_source_and_recent_filters_case_insensitively():
    result = classify_query_intent("LATEST notes FROM Readwise tagged #RAG tag:Graph")

    assert result["intent"] == "timeline"
    assert result["matched_cues"] == ["recent"]
    assert result["suggested_metadata_filters"] == {
        "tags": ["graph", "rag"],
        "source_project": ["readwise"],
        "date": {"relative": "recent"},
    }


def test_classify_query_intent_empty_query_is_low_confidence_exploratory():
    assert classify_query_intent(" \n\t ") == {
        "intent": "exploratory",
        "confidence": 0.1,
        "matched_cues": [],
        "suggested_search_mode": "semantic",
        "suggested_metadata_filters": {},
    }


def test_classify_query_intent_is_importable_from_graph_rag():
    from graph.rag import classify_query_intent as imported

    assert imported is classify_query_intent
