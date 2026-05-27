from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_metadata_completeness_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_metadata_completeness_groups_by_source_and_unknown():
    text = export_source_metadata_completeness_csv(
        [
            {"source_project": "web", "metadata": {"a": 1, "b": 2}},
            {"source_project": "web", "metadata": {}},
            {"metadata": {"a": 3}},
        ]
    )

    assert rows(text) == [
        {"source_project": "Unknown", "unit_count": "1", "units_with_metadata": "1", "metadata_coverage_ratio": "1.00", "unique_key_count": "1"},
        {"source_project": "web", "unit_count": "2", "units_with_metadata": "1", "metadata_coverage_ratio": "0.50", "unique_key_count": "2"},
    ]


def test_source_metadata_completeness_required_keys_and_path_mode(tmp_path):
    path = tmp_path / "metadata.csv"
    stats = export_source_metadata_completeness_csv(
        [{"source_project": "web", "metadata": {"a": 1}}],
        path,
        required_keys=["a", "b"],
    )

    assert rows(path.read_text(encoding="utf-8"))[0]["missing_required_keys"] == "b"
    assert rows(path.read_text(encoding="utf-8"))[0]["required_key_coverage_ratio"] == "0.50"
    assert stats["rows_exported"] == 1
