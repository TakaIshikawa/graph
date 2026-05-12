from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from graph.rag import score_source_reliability

NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)


@dataclass
class Result:
    id: str
    title: str
    source_project: str
    metadata: dict


def by_id(rows: list[dict], result_id: str) -> dict:
    return next(row for row in rows if row["result_id"] == result_id)


def test_source_reliability_scores_dicts_deterministically():
    rows = score_source_reliability(
        [
            {
                "id": "paper",
                "title": "Replication paper",
                "source_project": "papers",
                "citation_count": 25,
                "inbound_reference_count": 8,
                "author": "Ada Lovelace",
                "url": "https://journal.example/paper",
                "published_at": "2026-04-15",
            },
            {
                "id": "note",
                "title": "Local note",
                "source_project": "notes",
                "metadata": {"updated_at": "2023-01-01"},
            },
        ],
        now=NOW,
    )

    assert [row["result_id"] for row in rows] == ["paper", "note"]
    assert by_id(rows, "paper") == {
        "result_id": "paper",
        "title": "Replication paper",
        "score": 0.76,
        "grade": "B",
        "reasons": [
            "baseline reliability",
            "citation count 25",
            "inbound reference count 8",
            "author present",
            "domain present: journal.example",
            "recent timestamp",
            "common source project",
        ],
        "source_project": "papers",
        "domain": "journal.example",
        "timestamp": "2026-04-15T00:00:00+00:00",
    }
    assert by_id(rows, "note")["grade"] == "D"
    assert "missing author" in by_id(rows, "note")["reasons"]


def test_source_reliability_accepts_objects_and_tuple_payloads_consistently():
    object_result = Result(
        id="object",
        title="Object result",
        source_project="archive",
        metadata={
            "citation_count": 4,
            "inbound_reference_count": 2,
            "authors": ["Grace Hopper"],
            "source_url": "https://archive.example/item",
            "updated_at": "2026-03-01",
        },
    )
    dict_result = {
        "id": "dict",
        "title": "Dict result",
        "source_project": "archive",
        "metadata": {
            "citation_count": 4,
            "inbound_reference_count": 2,
            "authors": ["Grace Hopper"],
            "source_url": "https://archive.example/item",
            "updated_at": "2026-03-01",
        },
    }

    assert score_source_reliability([object_result], now=NOW)[0] == score_source_reliability(
        [(dict_result, 0.7)],
        now=NOW,
    )[0] | {"result_id": "object", "title": "Object result"}


def test_source_reliability_missing_metadata_produces_explanatory_reasons():
    rows = score_source_reliability([{}], now=NOW)

    assert rows == [
        {
            "result_id": "result-1",
            "title": None,
            "score": 0.08,
            "grade": "D",
            "reasons": [
                "baseline reliability",
                "missing citation count",
                "missing inbound references",
                "missing author",
                "missing domain",
                "missing usable timestamp",
                "missing source project",
            ],
            "source_project": "unknown",
            "domain": None,
            "timestamp": None,
        }
    ]


def test_source_reliability_limit_truncates_after_scoring_and_sorting():
    rows = score_source_reliability(
        [
            {"id": "low"},
            {
                "id": "high",
                "title": "High",
                "source_project": "papers",
                "domain": "example.org",
                "author": "A",
                "published_at": "2026-04-01",
                "citation_count": 5,
            },
        ],
        now=NOW,
        limit=1,
    )

    assert [row["result_id"] for row in rows] == ["high"]


@pytest.mark.parametrize("limit", [-1, True, "1"])
def test_source_reliability_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        score_source_reliability([], limit=limit)
