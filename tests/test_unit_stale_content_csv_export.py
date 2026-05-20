from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_stale_content_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=tags or [],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_stale_content_csv_flags_units_older_than_cutoff():
    text = export_unit_stale_content_csv(
        [
            unit("old", metadata={"last_seen": "2026-01-01"}, tags=["ai", "review"]),
            unit("fresh", metadata={"last_seen": "2026-05-01"}),
        ],
        cutoff_date="2026-04-01",
        reference_date="2026-05-10",
    )

    assert rows(text) == [
        {
            "unit_id": "old",
            "title": "Title old",
            "source": "Project",
            "last_seen_date": "2026-01-01",
            "age_days": "129",
            "tags": "ai; review",
            "stale_reason": "older_than_cutoff:2026-04-01",
        }
    ]


def test_unit_stale_content_csv_supports_max_age_days_and_reports_bad_dates():
    text = export_unit_stale_content_csv(
        [
            unit("old", metadata={"updated_at": "2026-04-01"}),
            unit("bad", metadata={"updated_at": "not a date"}),
            unit("missing"),
        ],
        max_age_days=30,
        reference_date="2026-05-10",
    )

    assert [row["unit_id"] for row in rows(text)] == ["old", "bad", "missing"]
    assert rows(text)[0]["stale_reason"] == "older_than_cutoff:2026-04-10"
    assert rows(text)[1]["stale_reason"] == "malformed_date"
    assert rows(text)[2]["stale_reason"] == "missing_date"


def test_unit_stale_content_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "stale.csv"
    units = [unit("old", metadata={"date": "2026-01-01"})]

    expected = export_unit_stale_content_csv(units, path=None, cutoff_date="2026-02-01", reference_date="2026-05-01")
    stats = export_unit_stale_content_csv(units, path, cutoff_date="2026-02-01", reference_date="2026-05-01")

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
    assert stats["cutoff_date"] == "2026-02-01"


@pytest.mark.parametrize("max_age_days", [-1, 1.2, True, "30"])
def test_unit_stale_content_csv_validates_max_age_days(max_age_days):
    with pytest.raises(ValueError, match="max_age_days must be a non-negative integer or None"):
        export_unit_stale_content_csv([], max_age_days=max_age_days)
