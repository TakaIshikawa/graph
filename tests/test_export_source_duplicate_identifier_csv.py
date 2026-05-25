from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_duplicate_identifier_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_duplicate_identifier_csv_emits_only_duplicate_normalized_identifiers():
    text = export_source_duplicate_identifier_csv(
        [
            {"id": "s1", "name": "Docs A", "url": "https://www.example.com/path/"},
            {"id": "s2", "name": "Docs B", "url": "https://example.com/path"},
            {"id": "s3", "name": "Unique", "doi": "10.1000/abc"},
        ]
    )

    assert rows(text) == [
        {
            "identifier_key": "url",
            "identifier_value": "https://example.com/path",
            "source_count": "2",
            "source_ids": "s1; s2",
            "source_names": "Docs A; Docs B",
            "collision_severity": "conflicting_names",
        }
    ]


def test_source_duplicate_identifier_csv_normalizes_doi_and_same_name_severity():
    text = export_source_duplicate_identifier_csv(
        [
            {"id": "a", "name": "Paper", "metadata": {"doi": "https://doi.org/10.555/XYZ"}},
            {"id": "b", "name": "Paper", "doi": "doi:10.555/xyz"},
        ]
    )

    assert rows(text)[0]["identifier_value"] == "10.555/xyz"
    assert rows(text)[0]["collision_severity"] == "same_name"
