from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_robots_policy_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_robots_policy_csv_reads_tolerant_aliases():
    text = export_source_robots_policy_csv(
        [
            {"id": "s2"},
            {"id": "s1", "metadata": {"allowed_by_robots": "yes", "robots_crawl_delay": 5, "policy_source": "stored", "checked_at": "2024-01-01T00:00:00Z"}},
        ]
    )

    assert rows(text) == [
        {
            "source_id": "s1",
            "robots_allowed": "true",
            "crawl_delay": "5",
            "robots_source": "stored",
            "policy_checked_at": "2024-01-01T00:00:00+00:00",
        },
        {
            "source_id": "s2",
            "robots_allowed": "unknown",
            "crawl_delay": "unknown",
            "robots_source": "unknown",
            "policy_checked_at": "unknown",
        },
    ]


def test_export_source_robots_policy_csv_path_mode(tmp_path):
    path = tmp_path / "robots.csv"
    stats = export_source_robots_policy_csv([{"id": "s1", "robots_allowed": False}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["robots_allowed"] == "false"
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
