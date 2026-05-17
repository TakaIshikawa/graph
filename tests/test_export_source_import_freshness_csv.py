from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export.source_import_freshness_csv import export_source_import_freshness_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, source_project: str, updated_at: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata={},
        tags=[],
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_import_freshness_csv_empty_input_has_header_only():
    assert export_source_import_freshness_csv([], now="2024-04-10") == (
        "source_project,unit_count,newest_updated_at,oldest_updated_at,days_since_newest,"
        "days_since_oldest,stale_unit_count,freshness_bucket\n"
    )


def test_source_import_freshness_csv_groups_sources_and_computes_age_buckets():
    text = export_source_import_freshness_csv(
        [
            unit("fresh-1", "Fresh", "2024-04-01"),
            unit("fresh-2", "Fresh", "2024-03-01"),
            unit("aging", "Aging", "2024-02-20"),
            unit("stale", "Stale", "2023-12-01"),
            {"id": "empty", "source_project": "Empty", "title": "No timestamp"},
        ],
        now="2024-04-10",
    )

    assert rows(text) == [
        {
            "source_project": "Aging",
            "unit_count": "1",
            "newest_updated_at": "2024-02-20",
            "oldest_updated_at": "2024-02-20",
            "days_since_newest": "50",
            "days_since_oldest": "50",
            "stale_unit_count": "0",
            "freshness_bucket": "aging",
        },
        {
            "source_project": "Empty",
            "unit_count": "1",
            "newest_updated_at": "",
            "oldest_updated_at": "",
            "days_since_newest": "",
            "days_since_oldest": "",
            "stale_unit_count": "0",
            "freshness_bucket": "empty",
        },
        {
            "source_project": "Fresh",
            "unit_count": "2",
            "newest_updated_at": "2024-04-01",
            "oldest_updated_at": "2024-03-01",
            "days_since_newest": "9",
            "days_since_oldest": "40",
            "stale_unit_count": "0",
            "freshness_bucket": "fresh",
        },
        {
            "source_project": "Stale",
            "unit_count": "1",
            "newest_updated_at": "2023-12-01",
            "oldest_updated_at": "2023-12-01",
            "days_since_newest": "131",
            "days_since_oldest": "131",
            "stale_unit_count": "1",
            "freshness_bucket": "stale",
        },
    ]


def test_source_import_freshness_csv_uses_updated_then_imported_fallback_and_path_mode(tmp_path):
    units = [
        unit("a", "Project A", datetime(2024, 4, 5, tzinfo=timezone.utc)),
        {"id": "b", "source_project": "Project A", "metadata": {"last_imported_at": "2024-01-01"}},
    ]
    path = tmp_path / "reports" / "freshness.csv"

    expected = export_source_import_freshness_csv(units, now=datetime(2024, 4, 10, tzinfo=timezone.utc))
    stats = export_source_import_freshness_csv(units, now="2024-04-10", path=path)

    assert rows(expected)[0]["newest_updated_at"] == "2024-04-05"
    assert rows(expected)[0]["oldest_updated_at"] == "2024-01-01"
    assert rows(expected)[0]["stale_unit_count"] == "1"
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "source_project_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
