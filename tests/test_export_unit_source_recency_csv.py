from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

import pytest

from graph.export import export_unit_source_recency_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    title: str | None = None,
    created_at: object = None,
    updated_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title if title is not None else f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_source_recency_empty_input_returns_header_only_csv():
    assert export_unit_source_recency_csv([], now=date(2024, 1, 10)) == (
        "unit_id,source_project,source_entity_type,title,best_date,age_days,recency_bucket\n"
    )


def test_unit_source_recency_prefers_updated_then_created_then_metadata():
    text = export_unit_source_recency_csv(
        [
            unit(
                "a",
                updated_at=datetime(2024, 1, 9, 12, 0, tzinfo=timezone.utc),
                created_at=date(2024, 1, 1),
                metadata={"published_date": "2023-01-01"},
            ),
            unit("b", created_at=date(2024, 1, 1), metadata={"date": "2024-01-09"}),
            unit("c", metadata={"published_date": "2023-01"}),
            unit("d", metadata={"date": "not a date"}),
        ],
        now=datetime(2024, 1, 10, tzinfo=timezone.utc),
        bucket_days=(1, 7, 30),
    )

    assert rows(text) == [
        {
            "unit_id": "d",
            "source_project": "max",
            "source_entity_type": "note",
            "title": "Title d",
            "best_date": "",
            "age_days": "",
            "recency_bucket": "undated",
        },
        {
            "unit_id": "c",
            "source_project": "max",
            "source_entity_type": "note",
            "title": "Title c",
            "best_date": "2023-01-01",
            "age_days": "374",
            "recency_bucket": "> 30 days",
        },
        {
            "unit_id": "b",
            "source_project": "max",
            "source_entity_type": "note",
            "title": "Title b",
            "best_date": "2024-01-01",
            "age_days": "9",
            "recency_bucket": "<= 30 days",
        },
        {
            "unit_id": "a",
            "source_project": "max",
            "source_entity_type": "note",
            "title": "Title a",
            "best_date": "2024-01-09T12:00:00+00:00",
            "age_days": "0",
            "recency_bucket": "<= 1 days",
        },
    ]


def test_unit_source_recency_unknown_fallbacks_and_metadata_year():
    text = export_unit_source_recency_csv(
        [unit("a", source_project=None, source_entity_type=None, title=None, metadata={"year": "2024"})],
        now=date(2024, 1, 10),
    )

    assert rows(text)[0] == {
        "unit_id": "a",
        "source_project": "Unknown",
        "source_entity_type": "Unknown",
        "title": "Title a",
        "best_date": "2024-01-01",
        "age_days": "9",
        "recency_bucket": "<= 30 days",
    }


def test_unit_source_recency_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "recency.csv"
    units = [unit("a", created_at=date(2024, 1, 1))]

    expected = export_unit_source_recency_csv(units, now=date(2024, 1, 2), bucket_days=(7, 30))
    stats = export_unit_source_recency_csv(units, path, now=date(2024, 1, 2), bucket_days=(7, 30))

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bucket_days": "7,30",
        "bytes_written": path.stat().st_size,
    }


def test_unit_source_recency_is_deterministic_for_reversed_input():
    units = [
        unit("c", source_project="Source B", created_at=date(2024, 1, 3)),
        unit("a", source_project="Source A", created_at=date(2024, 1, 1)),
        unit("b", source_project="Source A", created_at=date(2024, 1, 2)),
    ]

    assert export_unit_source_recency_csv(units, now=date(2024, 1, 10)) == export_unit_source_recency_csv(
        reversed(units), now=date(2024, 1, 10)
    )


@pytest.mark.parametrize("bucket_days", [(), (30, 7), (7, 7), (-1,), (True,), (1.5,)])
def test_unit_source_recency_validates_bucket_days(bucket_days):
    with pytest.raises(ValueError, match="bucket_days"):
        export_unit_source_recency_csv([], bucket_days=bucket_days)
