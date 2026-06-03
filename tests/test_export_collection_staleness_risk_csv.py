from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_collection_staleness_risk_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_staleness_risk_csv_aggregates_multi_collection_units():
    text = export_collection_staleness_risk_csv(
        [
            {"id": "a", "collections": ["work", "research"], "updated_at": "2026-01-01T00:00:00Z"},
            {"id": "b", "collections": ["work"], "updated_at": "2026-05-01T00:00:00Z"},
            {"id": "c", "metadata": {"collection": "personal"}, "updated_at": "2026-05-15T00:00:00Z"},
        ],
        stale_after_days=60,
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert rows(text) == [
        {"collection": "personal", "unit_count": "1", "oldest_updated": "2026-05-15T00:00:00+00:00", "newest_updated": "2026-05-15T00:00:00+00:00", "median_age_days": "17.0", "stale_unit_count": "0", "risk_level": "low"},
        {"collection": "research", "unit_count": "1", "oldest_updated": "2026-01-01T00:00:00+00:00", "newest_updated": "2026-01-01T00:00:00+00:00", "median_age_days": "151.0", "stale_unit_count": "1", "risk_level": "high"},
        {"collection": "work", "unit_count": "2", "oldest_updated": "2026-01-01T00:00:00+00:00", "newest_updated": "2026-05-01T00:00:00+00:00", "median_age_days": "91.0", "stale_unit_count": "1", "risk_level": "medium"},
    ]


def test_collection_staleness_risk_csv_path_mode_and_unassigned(tmp_path):
    path = tmp_path / "staleness.csv"
    units = [{"id": "a", "updated_at": "2026-01-01T00:00:00Z"}]

    expected = export_collection_staleness_risk_csv(units, now=datetime(2026, 1, 2, tzinfo=timezone.utc))
    stats = export_collection_staleness_risk_csv(units, path, now=datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert rows(expected)[0]["collection"] == "unassigned"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
