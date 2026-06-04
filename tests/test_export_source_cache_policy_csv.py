from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_cache_policy_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_cache_policy_csv_groups_normalized_policies_and_ttls():
    text = export_source_cache_policy_csv(
        [
            {"id": "a", "headers": {"cache-control": "no-store"}},
            {"id": "b", "metadata": {"cache_control": "max-age=60", "etag": "abc"}},
            {"id": "c", "ttl_seconds": 120, "last_modified": "2026-01-01"},
            {"id": "d", "metadata": {"cache_policy": "No Cache"}},
            {"id": "e", "metadata": {"immutable": True}},
            {"id": "f", "cache_control": "must-revalidate"},
            {"id": "g"},
        ]
    )

    by_policy = {row["cache_policy"]: row for row in rows(text)}
    assert by_policy["no-store"]["source_count"] == "1"
    assert by_policy["ttl"]["source_count"] == "2"
    assert by_policy["ttl"]["min_ttl_seconds"] == "60"
    assert by_policy["ttl"]["max_ttl_seconds"] == "120"
    assert by_policy["ttl"]["etag_count"] == "1"
    assert by_policy["ttl"]["last_modified_count"] == "1"
    assert {"immutable", "no-cache", "revalidate", "unknown"} <= set(by_policy)


def test_source_cache_policy_csv_path_mode_returns_write_metadata(tmp_path):
    path = tmp_path / "cache.csv"
    sources = [{"id": "a", "max_age": "max-age=30"}]

    expected = export_source_cache_policy_csv(sources)
    stats = export_source_cache_policy_csv(sources, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "source_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
