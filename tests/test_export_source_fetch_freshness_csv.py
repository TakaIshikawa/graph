from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_fetch_freshness_csv

NOW = datetime(2026, 5, 24, tzinfo=timezone.utc)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_fetch_freshness_csv_empty_input_returns_header():
    assert export_source_fetch_freshness_csv([], now=NOW) == "age_bucket,source_count,oldest_timestamp,newest_timestamp,source_ids\n"


def test_export_source_fetch_freshness_csv_uses_timestamp_precedence_and_buckets():
    text = export_source_fetch_freshness_csv(
        [
            {"id": "s1", "fetched_at": "2026-05-22T00:00:00+00:00", "updated_at": "2025-01-01T00:00:00+00:00"},
            {"id": "s2", "metadata": {"imported_at": "2026-05-01T00:00:00+00:00"}},
            {"id": "s3", "updated_at": "2026-03-01T00:00:00+00:00"},
        ],
        now=NOW,
    )

    assert rows(text) == [
        {
            "age_bucket": "0-7_days",
            "source_count": "1",
            "oldest_timestamp": "2026-05-22T00:00:00+00:00",
            "newest_timestamp": "2026-05-22T00:00:00+00:00",
            "source_ids": "s1",
        },
        {
            "age_bucket": "8-30_days",
            "source_count": "1",
            "oldest_timestamp": "2026-05-01T00:00:00+00:00",
            "newest_timestamp": "2026-05-01T00:00:00+00:00",
            "source_ids": "s2",
        },
        {
            "age_bucket": "31-90_days",
            "source_count": "1",
            "oldest_timestamp": "2026-03-01T00:00:00+00:00",
            "newest_timestamp": "2026-03-01T00:00:00+00:00",
            "source_ids": "s3",
        },
    ]


def test_export_source_fetch_freshness_csv_unknown_for_missing_or_invalid_path_mode(tmp_path):
    path = tmp_path / "freshness.csv"
    stats = export_source_fetch_freshness_csv([{"id": "s1", "fetched_at": "not-a-date"}], path, now=NOW)

    assert rows(path.read_text(encoding="utf-8"))[0]["age_bucket"] == "unknown"
    assert stats["source_count"] == 1
