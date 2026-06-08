from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_quality_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_quality_csv_groups_missing_quality_signals_by_source():
    text = export_source_quality_csv(
        [
            {"id": "a", "source_project": "A", "title": "Title", "content": "content", "metadata": {"x": 1}, "tags": ["tag"]},
            {"id": "b", "source_project": "A", "title": "", "content": "", "metadata": {}, "tags": []},
            {"id": "c", "source_project": "", "title": "Unknown", "content": "content", "metadata": {"x": 1}, "tags": ["tag"]},
        ]
    )

    assert rows(text) == [
        {
            "source_project": "A",
            "unit_count": "2",
            "missing_title_count": "1",
            "missing_content_count": "1",
            "missing_metadata_count": "1",
            "missing_tags_count": "1",
            "complete_unit_count": "1",
            "quality_score": "0.5",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "missing_title_count": "0",
            "missing_content_count": "0",
            "missing_metadata_count": "0",
            "missing_tags_count": "0",
            "complete_unit_count": "1",
            "quality_score": "1.0",
        },
    ]


def test_source_quality_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "quality.csv"
    units = [{"id": "a", "source_project": "A"}]

    expected = export_source_quality_csv(units)
    stats = export_source_quality_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "source_count": 1, "bytes_written": path.stat().st_size}
