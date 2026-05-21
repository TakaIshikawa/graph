from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_metadata_entropy_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_metadata_entropy_csv_counts_scalar_and_list_values():
    text = export_unit_metadata_entropy_csv(
        [
            {"metadata": {"status": "open", "tags": ["a", "b"]}},
            {"metadata": {"status": "open", "tags": ["a"]}},
            {"metadata": {"status": "closed"}},
        ]
    )

    result = {row["metadata_key"]: row for row in rows(text)}
    assert result["status"]["unit_count"] == "3"
    assert result["status"]["distinct_value_count"] == "2"
    assert result["status"]["top_value"] == "open"
    assert result["tags"]["non_empty_count"] == "2"
    assert result["tags"]["distinct_value_count"] == "2"


def test_export_unit_metadata_entropy_csv_path_mode(tmp_path):
    path = tmp_path / "entropy.csv"
    stats = export_unit_metadata_entropy_csv([{"metadata": {"status": "open"}}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["metadata_key"] == "status"
    assert stats["rows_exported"] == 1
