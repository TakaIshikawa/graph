from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_metadata_key_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_metadata_key_coverage_counts_against_all_units():
    result = rows(
        export_units_to_metadata_key_coverage_csv(
            [
                {"id": "b", "metadata": {"source": "web", "tags": ["x"]}},
                {"id": "a", "metadata": {"source": "file", "pages": 3}},
                {"id": "c", "metadata": {}},
            ]
        )
    )

    assert result[0]["metadata_key"] == "source"
    assert result[0]["unit_count"] == "2"
    assert result[0]["coverage_percent"] == "66.67"
    assert result[0]["sample_unit_ids"] == "a; b"
    assert result[0]["value_type_mix"] == "string:2"
    assert result[1]["metadata_key"] == "pages"
    assert result[2]["metadata_key"] == "tags"


def test_metadata_key_coverage_samples_are_capped():
    units = [{"id": f"u{i}", "metadata": {"key": i}} for i in range(7)]

    result = rows(export_units_to_metadata_key_coverage_csv(units))[0]

    assert result["sample_unit_ids"] == "u0; u1; u2; u3; u4"
    assert result["value_type_mix"] == "number:7"


def test_metadata_key_coverage_empty_metadata_header_only():
    assert export_units_to_metadata_key_coverage_csv([{"id": "u1", "metadata": {}}, {"id": "u2"}]) == (
        "metadata_key,unit_count,coverage_percent,sample_unit_ids,value_type_mix\n"
    )
