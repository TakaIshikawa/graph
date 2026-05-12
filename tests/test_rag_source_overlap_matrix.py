from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.source_overlap_matrix import build_source_overlap_matrix


@dataclass
class Result:
    source_project: str
    metadata: dict


def test_source_overlap_matrix_compares_sources_by_shared_values():
    rows = build_source_overlap_matrix(
        [
            {"source_project": "notes", "tags": ["Battery", "Grid"], "metadata": {"authors": ["Ada"]}},
            {"source_project": "web", "metadata": {"tags": ["battery"], "topics": ["Storage"], "authors": ["Ada"]}},
            Result("docs", {"entities": ["Grid"], "urls": [{"url": "https://example.test/a"}]}),
        ]
    )

    assert rows == [
        {
            "source_a": "notes",
            "source_b": "web",
            "overlap_count": 2,
            "shared_values": ["ada", "battery"],
            "source_a_count": 3,
            "source_b_count": 3,
            "jaccard": 0.5,
        },
        {
            "source_a": "docs",
            "source_b": "notes",
            "overlap_count": 1,
            "shared_values": ["grid"],
            "source_a_count": 2,
            "source_b_count": 3,
            "jaccard": 0.25,
        },
    ]


def test_unknown_source_and_min_overlap_are_deterministic():
    rows = build_source_overlap_matrix(
        [
            {"tags": ["alpha"]},
            {"source_project": "known", "metadata": {"tags": ["alpha", "beta"]}},
        ],
        min_overlap=1,
    )

    assert rows[0]["source_a"] == "known"
    assert rows[0]["source_b"] == "unknown"
    assert rows[0]["shared_values"] == ["alpha"]
    assert build_source_overlap_matrix(rows, min_overlap=2) == []


@pytest.mark.parametrize("min_overlap", [-1, 1.1, True, "1"])
def test_source_overlap_matrix_validates_min_overlap(min_overlap):
    with pytest.raises(ValueError, match="min_overlap must be a non-negative integer"):
        build_source_overlap_matrix([], min_overlap=min_overlap)
