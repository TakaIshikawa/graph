from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_identifier_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_identifier_coverage_empty_input_has_header():
    assert export_source_identifier_coverage_csv([]) == (
        "source_id,source_name,identifier_count,identifiers_present,identifiers_missing,coverage_score\n"
    )


def test_source_identifier_coverage_counts_present_and_missing_defaults():
    result = rows(
        export_source_identifier_coverage_csv(
            [
                {
                    "id": "s1",
                    "name": "Source",
                    "url": "https://example.test",
                    "metadata": {"external_id": "e1", "feed_url": ""},
                }
            ]
        )
    )[0]

    assert result["identifier_count"] == "2"
    assert result["identifiers_present"] == "url;external_id"
    assert "feed_url" in result["identifiers_missing"]
    assert result["coverage_score"] == "0.29"


def test_source_identifier_coverage_supports_custom_keys_and_list_values(tmp_path):
    path = tmp_path / "ids.csv"
    stats = export_source_identifier_coverage_csv(
        [{"source_id": "s1", "metadata": {"aliases": ["a1"], "blank": []}}],
        path,
        identifier_keys=("aliases", "blank"),
    )

    result = rows(path.read_text(encoding="utf-8"))[0]
    assert result["identifiers_present"] == "aliases"
    assert result["identifiers_missing"] == "blank"
    assert stats["rows_exported"] == 1
