from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_tag_vocabulary_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: object = "alpha",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> KnowledgeUnit:
    timestamp = created_at or datetime(2026, 5, 1, tzinfo=timezone.utc)
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        tags=tags or [],
        metadata=metadata or {},
        created_at=timestamp,
        ingested_at=timestamp,
        updated_at=timestamp,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_tag_vocabulary_csv_groups_case_variants_by_source():
    text = export_source_tag_vocabulary_csv(
        [
            unit("u2", tags=["Solar Storage"], metadata={"observed_date": "2026-05-03"}),
            unit("u1", tags=["solar-storage", "Grid"], metadata={"observed_date": "2026-05-01"}),
            unit("u3", source_project="beta", tags=["Solar storage"], metadata={"observed_date": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "alpha",
            "normalized_tag": "grid",
            "raw_tag_variants": "Grid",
            "unit_count": "1",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-01",
            "representative_unit_ids": "u1",
        },
        {
            "source_project": "alpha",
            "normalized_tag": "solar storage",
            "raw_tag_variants": "Solar Storage; solar-storage",
            "unit_count": "2",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-03",
            "representative_unit_ids": "u1; u2",
        },
        {
            "source_project": "beta",
            "normalized_tag": "solar storage",
            "raw_tag_variants": "Solar storage",
            "unit_count": "1",
            "first_seen": "2026-05-02",
            "last_seen": "2026-05-02",
            "representative_unit_ids": "u3",
        },
    ]


def test_export_source_tag_vocabulary_csv_uses_unknown_source_and_metadata_tags():
    text = export_source_tag_vocabulary_csv(
        [{"id": "u1", "metadata": {"tags": ["Research"], "date": "2026-05-04"}}]
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "normalized_tag": "research",
            "raw_tag_variants": "Research",
            "unit_count": "1",
            "first_seen": "2026-05-04",
            "last_seen": "2026-05-04",
            "representative_unit_ids": "u1",
        }
    ]


def test_export_source_tag_vocabulary_csv_path_and_file_like_output(tmp_path):
    units = [unit("u1", tags=["Tag"])]
    expected = export_source_tag_vocabulary_csv(units)
    path = tmp_path / "source-tags.csv"

    stats = export_source_tag_vocabulary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["tag_rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size

    buffer = StringIO()
    file_stats = export_source_tag_vocabulary_csv(units, buffer)
    assert buffer.getvalue() == expected
    assert file_stats["bytes_written"] == len(expected)
