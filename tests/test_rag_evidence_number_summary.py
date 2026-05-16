from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.evidence_number_summary import summarize_evidence_numbers


@dataclass
class Result:
    source_id: str
    content: str = ""
    metadata: dict | None = None


def test_summarize_evidence_numbers_classifies_numeric_fact_kinds_in_order():
    rows = summarize_evidence_numbers(
        [
            {
                "id": "a",
                "content": "On 2024-05-01 revenue reached $1,200.50. Adoption rose 12.5% in 2025 with 42 pilots.",
            }
        ],
        "revenue adoption pilots",
    )

    assert [(row["value"], row["kind"], row["matched_query_terms"]) for row in rows] == [
        ("2024-05-01", "date", ["revenue"]),
        ("$1,200.50", "currency", ["revenue"]),
        ("12.5%", "percent", ["adoption", "pilots"]),
        ("2025", "year", ["adoption", "pilots"]),
        ("42", "number", ["adoption", "pilots"]),
    ]
    assert rows[0]["context"] == "On 2024-05-01 revenue reached $1,200.50."
    assert [row["position"] for row in rows] == [0, 1, 2, 3, 4]


def test_summarize_evidence_numbers_supports_objects_tuples_and_metadata_fallbacks():
    rows = summarize_evidence_numbers(
        [
            (Result("object", content="Object estimate was 30%."), 0.8),
            {"metadata": {"unit_id": "meta", "snippet": "Metadata budget was 500 USD on 01/02/2024."}},
        ]
    )

    assert [(row["result_id"], row["value"], row["kind"]) for row in rows] == [
        ("object", "30%", "percent"),
        ("meta", "500 USD", "currency"),
        ("meta", "01/02/2024", "date"),
    ]


def test_summarize_evidence_numbers_handles_empty_query_limit_and_no_matches():
    assert summarize_evidence_numbers([{"content": "No numbers here."}]) == []
    assert summarize_evidence_numbers([{"content": "There are 2 matches and 3 backups."}], max_numbers=1) == [
        {
            "result_id": "result-1",
            "value": "2",
            "kind": "number",
            "context": "There are 2 matches and 3 backups.",
            "matched_query_terms": [],
            "position": 0,
        }
    ]
    assert summarize_evidence_numbers([{"content": "There are 2 matches."}], max_numbers=0) == []


@pytest.mark.parametrize("max_numbers", [-1, 1.2, True, "4"])
def test_summarize_evidence_numbers_validates_max_numbers(max_numbers):
    with pytest.raises(ValueError, match="max_numbers"):
        summarize_evidence_numbers([], max_numbers=max_numbers)
