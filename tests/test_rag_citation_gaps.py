from __future__ import annotations

from dataclasses import dataclass

from graph.rag.citation_gaps import prioritize_citation_gaps


@dataclass
class Result:
    id: str
    title: str
    content: str
    source_project: str
    tags: list[str]
    metadata: dict


def test_prioritize_citation_gaps_ranks_missing_signals_deterministically():
    rows = prioritize_citation_gaps(
        [
            {
                "id": "cited",
                "title": "Cited battery note",
                "source_project": "papers",
                "metadata": {"doi": "10.1000/example", "url": "https://example.com"},
            },
            {
                "id": "missing",
                "title": "Battery market claim",
                "source_project": "notes",
                "content": "Market storage claim.",
                "tags": ["storage"],
                "metadata": {},
            },
            {
                "id": "partial",
                "title": "Storage brief",
                "source_project": "briefs",
                "metadata": {"references": ["Smith 2025"]},
            },
        ],
        query="battery storage",
    )

    assert [row["result_id"] for row in rows] == ["missing", "partial", "cited"]
    assert rows[0] == {
        "result_id": "missing",
        "title": "Battery market claim",
        "source_project": "notes",
        "citation_signal_count": 0,
        "missing_signal_reasons": [
            "missing citation signals: citations, references, url, source_url, doi, isbn, cited_by",
            "no citation metadata found",
            "matches query terms: battery, storage",
        ],
        "matched_query_terms": ["battery", "storage"],
        "priority_score": 8.0,
    }
    assert rows[-1]["citation_signal_count"] == 2
    assert rows[-1]["priority_score"] == 5.5


def test_prioritize_citation_gaps_accepts_object_results_and_metadata_signals():
    rows = prioritize_citation_gaps(
        [
            Result(
                id="object-a",
                title="Uncited climate finding",
                content="Climate evidence summary.",
                source_project="lab",
                tags=["climate"],
                metadata={"cited_by": 4},
            ),
            Result(
                id="object-b",
                title="Book evidence",
                content="",
                source_project="library",
                tags=[],
                metadata={"isbn": "9780000000000", "citations": ["Catalog"]},
            ),
        ],
        query="climate",
    )

    assert rows[0]["result_id"] == "object-a"
    assert rows[0]["citation_signal_count"] == 1
    assert rows[0]["matched_query_terms"] == ["climate"]
    assert rows[1]["citation_signal_count"] == 2


def test_query_terms_only_boost_matching_results():
    rows = prioritize_citation_gaps(
        [
            {"id": "alpha", "title": "Alpha", "metadata": {}},
            {"id": "beta", "title": "Beta", "metadata": {}},
        ],
        query="beta unmatched",
    )

    assert [row["result_id"] for row in rows] == ["beta", "alpha"]
    assert rows[0]["matched_query_terms"] == ["beta"]
    assert rows[1]["matched_query_terms"] == []
