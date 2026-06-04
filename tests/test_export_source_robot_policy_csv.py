from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_source_robot_policy_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_robot_policy_csv_handles_objects_metadata_and_sorting():
    text = export_source_robot_policy_csv(
        [
            {"id": "b", "name": "Beta", "robots_allowed": "yes", "crawl_delay": 5, "user_agent": "*", "policy_url": "https://b.test/robots.txt", "checked_at": "2026-01-02T00:00:00Z"},
            SimpleNamespace(id="a", metadata={"source_name": "Alpha", "allowed_by_robots": "no", "robots_crawl_delay": "10", "robots_user_agent": "GraphBot", "robots_url": "https://a.test/robots.txt", "robots_checked_at": "2026-01-01"}),
        ]
    )

    assert rows(text) == [
        {"source_id": "a", "source_name": "Alpha", "robots_allowed": "false", "crawl_delay": "10", "user_agent": "GraphBot", "policy_url": "https://a.test/robots.txt", "checked_at": "2026-01-01T00:00:00+00:00"},
        {"source_id": "b", "source_name": "Beta", "robots_allowed": "true", "crawl_delay": "5", "user_agent": "*", "policy_url": "https://b.test/robots.txt", "checked_at": "2026-01-02T00:00:00+00:00"},
    ]


def test_source_robot_policy_csv_path_mode_returns_stats(tmp_path):
    path = tmp_path / "robots.csv"
    stats = export_source_robot_policy_csv([{"id": "s"}], path)

    assert stats["path"] == str(path)
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
