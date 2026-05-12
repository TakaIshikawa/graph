from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from graph.rag.evidence_freshness import score_evidence_freshness
from graph.types.models import KnowledgeUnit


NOW = datetime(2026, 5, 12, tzinfo=timezone.utc)


def test_score_evidence_freshness_supports_mapping_results_and_metadata_dates():
    rows = score_evidence_freshness(
        [
            {"id": "old", "source_project": "notes", "created_at": "2025-01-01T00:00:00Z"},
            {
                "id": "fresh",
                "metadata": {
                    "source_project": "web",
                    "published_at": "2026-05-10",
                },
            },
            {"id": "undated", "source_project": "notes"},
        ],
        now=NOW,
    )

    assert [row["result_id"] for row in rows] == ["fresh", "old", "undated"]
    assert rows[0] == {
        "result_id": "fresh",
        "source_project": "web",
        "freshest_date": "2026-05-10T00:00:00+00:00",
        "age_days": 2,
        "freshness_bucket": "fresh",
        "freshness_score": 1.0,
    }
    assert rows[1]["freshness_bucket"] == "stale"
    assert rows[2]["freshness_bucket"] == "undated"
    assert rows[2]["freshness_score"] == 0.0


def test_score_evidence_freshness_supports_object_results_and_nested_units():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="source-1",
        source_entity_type="highlight",
        title="Highlight",
        content="Text",
        metadata={"updated_at": "2026-04-01T00:00:00Z"},
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    result = SimpleNamespace(id="wrapper", source_project="search", unit=unit)

    rows = score_evidence_freshness([result], now=NOW)

    assert rows == [
        {
            "result_id": "wrapper",
            "source_project": "search",
            "freshest_date": "2026-04-01T00:00:00+00:00",
            "age_days": 41,
            "freshness_bucket": "recent",
            "freshness_score": 0.75,
        }
    ]


def test_score_evidence_freshness_uses_freshest_date_and_sorts_by_score_then_id():
    rows = score_evidence_freshness(
        [
            {"id": "b", "source_project": "notes", "created_at": "2026-01-01", "updated_at": "2026-02-01"},
            {"id": "a", "source_project": "notes", "date": "2026-02-01"},
            {"id": "c", "source_project": "notes", "published_at": "not-a-date"},
        ],
        now=NOW,
    )

    assert [row["result_id"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["freshest_date"] == "2026-02-01T00:00:00+00:00"
    assert rows[0]["freshness_score"] == rows[1]["freshness_score"]
