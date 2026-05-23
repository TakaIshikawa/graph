from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_import_error_summary_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_import_error_summary_empty_input_has_header():
    assert export_source_import_error_summary_csv([]) == (
        "source_id,source_name,error_count,warning_count,last_error_at,last_error_message\n"
    )


def test_source_import_error_summary_counts_errors_and_warnings_and_sorts():
    text = export_source_import_error_summary_csv(
        [
            {"id": "b", "name": "Beta", "metadata": {"errors": ["bad"], "warnings": ["soft"]}},
            {
                "id": "a",
                "name": "Alpha",
                "metadata": {
                    "error_count": 2,
                    "warning_count": "3",
                    "last_error_message": "failed",
                },
            },
        ]
    )

    result = rows(text)
    assert [row["source_id"] for row in result] == ["a", "b"]
    assert result[0]["error_count"] == "2"
    assert result[0]["warning_count"] == "3"
    assert result[0]["last_error_message"] == "failed"


def test_source_import_error_summary_uses_recent_error_record_and_path_mode(tmp_path):
    path = tmp_path / "errors.csv"
    sources = [
        {
            "source_id": "s1",
            "title": "Source",
            "metadata": {
                "import_errors": [
                    {"message": "old", "at": "2024-01-01T00:00:00Z"},
                    {"message": "new", "at": "2024-01-03T00:00:00Z"},
                ]
            },
        }
    ]

    stats = export_source_import_error_summary_csv(sources, path)
    result = rows(path.read_text(encoding="utf-8"))[0]
    assert result["last_error_at"] == "2024-01-03T00:00:00+00:00"
    assert result["last_error_message"] == "new"
    assert stats["source_count"] == 1
