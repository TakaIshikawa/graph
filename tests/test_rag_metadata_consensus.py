from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.metadata_consensus import summarize_metadata_consensus


@dataclass
class Result:
    id: str
    metadata: dict
    source_project: str


def test_metadata_consensus_reports_supported_values_case_insensitively():
    rows = summarize_metadata_consensus(
        [
            {"id": "a", "metadata": {"status": "Confirmed", "sentiment": "Positive"}, "source_project": "one"},
            {"id": "b", "status": "confirmed", "source_project": "two"},
            Result("c", {"status": "Draft", "sentiment": "positive"}, "one"),
        ]
    )

    assert rows == [
        {
            "field": "sentiment",
            "value": "Positive",
            "support_count": 2,
            "support_ratio": 0.666667,
            "representative_results": ["a", "c"],
            "source_projects": ["one"],
        },
        {
            "field": "status",
            "value": "Confirmed",
            "support_count": 2,
            "support_ratio": 0.666667,
            "representative_results": ["a", "b"],
            "source_projects": ["one", "two"],
        },
    ]


def test_metadata_consensus_supports_custom_fields_and_min_support():
    rows = summarize_metadata_consensus(
        [
            {"id": "1", "metadata": {"phase": "alpha"}},
            {"id": "2", "phase": "alpha"},
            {"id": "3", "phase": "alpha"},
        ],
        fields=["phase"],
        min_support=3,
    )

    assert rows[0]["field"] == "phase"
    assert rows[0]["support_count"] == 3


@pytest.mark.parametrize("min_support", [0, -1, 1.2, True, "2"])
def test_metadata_consensus_validates_min_support(min_support):
    with pytest.raises(ValueError, match="min_support must be a positive integer"):
        summarize_metadata_consensus([], min_support=min_support)
