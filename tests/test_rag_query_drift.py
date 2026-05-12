from __future__ import annotations

from graph.rag import analyze_query_drift


def test_query_drift_scores_results_by_missing_focus_terms():
    rows = analyze_query_drift(
        "battery storage policy",
        [
            {
                "id": "on-topic",
                "title": "Battery storage policy",
                "snippet": "Grid storage rules.",
                "tags": ["energy"],
            },
            {
                "id": "partial",
                "title": "Battery market",
                "snippet": "Cells and manufacturing.",
            },
            {
                "id": "drifted",
                "title": "Cloud database",
                "snippet": "Index maintenance.",
            },
        ],
    )

    assert [row["result_id"] for row in rows] == ["drifted", "partial", "on-topic"]
    assert rows[0] == {
        "result_id": "drifted",
        "title": "Cloud database",
        "matched_terms": [],
        "missing_terms": ["battery", "policy", "storage"],
        "drift_score": 1.0,
    }
    assert rows[-1]["matched_terms"] == ["battery", "policy", "storage"]
    assert rows[-1]["drift_score"] == 0.0


def test_query_drift_matches_title_snippet_tags_and_metadata():
    rows = analyze_query_drift(
        "retrieval citation graph",
        [
            {
                "id": "metadata-match",
                "title": "Search notes",
                "snippet": "Local ranking.",
                "tags": ["Graph"],
                "metadata": {"keywords": [{"keyword": "Citation"}, "retrieval"]},
            }
        ],
    )

    assert rows == [
        {
            "result_id": "metadata-match",
            "title": "Search notes",
            "matched_terms": ["citation", "graph", "retrieval"],
            "missing_terms": [],
            "drift_score": 0.0,
        }
    ]


def test_query_drift_normalizes_stopwords_and_repeated_terms():
    rows = analyze_query_drift(
        "What is battery battery storage and storage?",
        [
            {"id": "storage", "title": "Storage"},
            {"id": "battery", "title": "Battery"},
        ],
    )

    assert rows == [
        {
            "result_id": "battery",
            "title": "Battery",
            "matched_terms": ["battery"],
            "missing_terms": ["storage"],
            "drift_score": 0.5,
        },
        {
            "result_id": "storage",
            "title": "Storage",
            "matched_terms": ["storage"],
            "missing_terms": ["battery"],
            "drift_score": 0.5,
        },
    ]


def test_query_drift_empty_query_and_stable_sorting():
    results = [
        {"id": "b", "title": "Beta"},
        {"id": "a", "title": "Alpha"},
    ]

    assert analyze_query_drift("", []) == []
    assert analyze_query_drift("", results) == analyze_query_drift("", reversed(results))
    assert [row["result_id"] for row in analyze_query_drift("", results)] == ["a", "b"]
