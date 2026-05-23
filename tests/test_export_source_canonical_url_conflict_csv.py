from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_canonical_url_conflict_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_canonical_url_conflict_csv_empty_input_returns_header():
    assert export_source_canonical_url_conflict_csv([]) == "conflict_key,source_count,raw_urls,source_ids\n"


def test_export_source_canonical_url_conflict_csv_groups_normalized_duplicates():
    text = export_source_canonical_url_conflict_csv(
        [
            {"id": "s1", "canonical_url": "https://Example.test/page?ref=1"},
            {"id": "s2", "url": "https://example.test/page#section"},
            {"id": "s3", "url": "https://other.test/page"},
        ]
    )

    assert rows(text) == [
            {
                "conflict_key": "https://example.test/page",
                "source_count": "2",
                "raw_urls": "https://example.test/page#section; https://Example.test/page?ref=1",
                "source_ids": "s1; s2",
            }
        ]


def test_export_source_canonical_url_conflict_csv_reports_one_source_conflicting_raw_urls_path_mode(tmp_path):
    path = tmp_path / "canonical-conflicts.csv"
    stats = export_source_canonical_url_conflict_csv(
        [{"id": "s1", "metadata": {"canonical_url": ["https://example.test/a", "https://example.test/a/"]}}],
        path,
    )

    assert rows(path.read_text(encoding="utf-8"))[0]["source_ids"] == "s1"
    assert stats["rows_exported"] == 1
