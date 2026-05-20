from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_tag_first_seen_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str,
    tags: list[str],
    source_project: str = "Project",
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="content",
        metadata=metadata or {},
        tags=tags,
        created_at=created_at,
        updated_at=created_at,
        ingested_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_first_seen_csv_reports_earliest_unit_per_tag():
    text = export_tag_first_seen_csv(
        [
            unit("b", title="Beta", tags=["ai"], metadata={"published_at": "2026-05-03"}),
            unit("a", title="Alpha", tags=["ai", "ml"], source_project="Source A", metadata={"source_date": "2026-05-01T12:00:00Z"}),
            unit("c", title="Gamma", tags=["ml"], metadata={"published_at": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {
            "tag": "ai",
            "first_seen": "2026-05-01",
            "unit_id": "a",
            "unit_title": "Alpha",
            "source": "Source A",
            "total_units_with_tag": "2",
        },
        {
            "tag": "ml",
            "first_seen": "2026-05-01",
            "unit_id": "a",
            "unit_title": "Alpha",
            "source": "Source A",
            "total_units_with_tag": "2",
        },
    ]


def test_tag_first_seen_csv_uses_stable_tie_breaks_and_keeps_undated_tags_last():
    text = export_tag_first_seen_csv(
        [
            unit("b", title="Beta", tags=["tie"], metadata={"date": "2026-05-01"}),
            unit("a", title="Alpha", tags=["tie"], metadata={"date": "2026-05-01"}),
            unit("z", title="Undated", tags=["undated"], created_at=None),
        ]
    )

    assert rows(text) == [
        {
            "tag": "tie",
            "first_seen": "2026-05-01",
            "unit_id": "a",
            "unit_title": "Alpha",
            "source": "Project",
            "total_units_with_tag": "2",
        },
        {
            "tag": "undated",
            "first_seen": "",
            "unit_id": "z",
            "unit_title": "Undated",
            "source": "Project",
            "total_units_with_tag": "1",
        },
    ]


def test_tag_first_seen_csv_falls_back_to_unit_dates_and_writes_path(tmp_path):
    path = tmp_path / "tags.csv"
    units = [unit("a", title="Alpha", tags=["ai"], created_at=datetime(2026, 5, 4, tzinfo=timezone.utc))]

    expected = export_tag_first_seen_csv(units)
    stats = export_tag_first_seen_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert rows(expected)[0]["first_seen"] == "2026-05-04"
    assert stats["rows_exported"] == 1
