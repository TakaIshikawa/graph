from __future__ import annotations

from graph.rag.query_exclusion_criteria import detect_query_exclusion_criteria


def test_detect_query_exclusion_criteria_keeps_multiple_cues_in_order():
    result = detect_query_exclusion_criteria(
        "Find CRM options without Salesforce, exclude beta products, and not from vendor blogs."
    )

    assert result["has_exclusions"] is True
    assert result["exclusions"] == [
        {"cue": "without", "text": "Salesforce", "span": [17, 35]},
        {"cue": "exclude", "text": "beta products", "span": [37, 58]},
        {"cue": "not from", "text": "vendor blogs", "span": [64, 85]},
    ]
    assert result["cue_counts"] == {"without": 1, "exclude": 1, "not from": 1}


def test_detect_query_exclusion_criteria_handles_empty_and_cue_free_queries():
    assert detect_query_exclusion_criteria("") == {
        "has_exclusions": False,
        "exclusions": [],
        "cue_counts": {},
        "normalized_query_without_exclusions": "",
    }
    assert detect_query_exclusion_criteria("Summarize retrieval quality evidence.") == {
        "has_exclusions": False,
        "exclusions": [],
        "cue_counts": {},
        "normalized_query_without_exclusions": "Summarize retrieval quality evidence.",
    }


def test_detect_query_exclusion_criteria_normalizes_query_without_destroying_remainder():
    result = detect_query_exclusion_criteria(
        "Compare vector databases do not include archived projects while preserving license notes."
    )

    assert result["exclusions"][0]["text"] == "archived projects"
    assert result["normalized_query_without_exclusions"] == (
        "Compare vector databases while preserving license notes."
    )
