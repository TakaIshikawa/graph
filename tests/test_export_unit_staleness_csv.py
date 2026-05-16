from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

import pytest

from graph.export import export_unit_staleness_csv
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, tags: list[str] | None = None) -> KnowledgeUnit:
    fallback = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return KnowledgeUnit(
        id=unit_id,
        source_project="alpha",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        tags=tags or [],
        metadata=metadata or {},
        created_at=fallback,
        ingested_at=fallback,
        updated_at=fallback,
    )


def edge(edge_id: str, source: str, target: str, *, metadata: dict | None = None) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=EdgeRelation.RELATES_TO,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_staleness_uses_documented_date_precedence_and_relation_context():
    text = export_unit_staleness_csv(
        [
            unit(
                "a",
                metadata={
                    "observed_date": "2026-05-01",
                    "updated_at": "2026-04-01",
                    "created_at": "2025-01-01",
                    "source_date": "2024-01-01",
                },
                tags=["Research", "research"],
            ),
            unit("b", metadata={"updated_at": "2026-02-01"}),
            unit("c", metadata={"created_at": "2025-05-01"}),
            unit("d", metadata={"source_date": "2024-01-01"}),
        ],
        [
            edge("recent", "a", "b", metadata={"observed_date": "2026-05-10"}),
            edge("old", "c", "d", metadata={"observed_date": "2025-01-01"}),
        ],
        reference_date="2026-05-16",
    )

    assert rows(text) == [
        {
            "unit_id": "d",
            "title": "Title d",
            "source_project": "alpha",
            "last_activity_date": "2024-01-01",
            "age_days": "866",
            "staleness_bucket": "dormant",
            "has_recent_relations": "false",
            "relation_count": "1",
            "tags": "",
        },
        {
            "unit_id": "c",
            "title": "Title c",
            "source_project": "alpha",
            "last_activity_date": "2025-05-01",
            "age_days": "380",
            "staleness_bucket": "dormant",
            "has_recent_relations": "false",
            "relation_count": "1",
            "tags": "",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "alpha",
            "last_activity_date": "2026-02-01",
            "age_days": "104",
            "staleness_bucket": "stale",
            "has_recent_relations": "true",
            "relation_count": "1",
            "tags": "",
        },
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "alpha",
            "last_activity_date": "2026-05-01",
            "age_days": "15",
            "staleness_bucket": "current",
            "has_recent_relations": "true",
            "relation_count": "1",
            "tags": "Research; research",
        },
    ]


def test_unit_staleness_handles_dict_units_missing_dates_and_unknown_source():
    text = export_unit_staleness_csv(
        [{"id": "x", "title": "Loose", "tags": ["Tag"]}],
        reference_date="2026-05-16",
    )

    assert rows(text) == [
        {
            "unit_id": "x",
            "title": "Loose",
            "source_project": "Unknown",
            "last_activity_date": "",
            "age_days": "",
            "staleness_bucket": "missing_date",
            "has_recent_relations": "false",
            "relation_count": "0",
            "tags": "Tag",
        }
    ]


def test_unit_staleness_path_mode_and_validation(tmp_path):
    units = [unit("a", metadata={"observed_date": "2026-05-01"})]
    expected = export_unit_staleness_csv(units, reference_date="2026-05-16")
    path = tmp_path / "staleness.csv"

    stats = export_unit_staleness_csv(units, path=path, reference_date="2026-05-16")

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["reference_date"] == "2026-05-16"
    assert stats["bytes_written"] == path.stat().st_size

    with pytest.raises(ValueError):
        export_unit_staleness_csv(units, reference_date="not-a-date")
