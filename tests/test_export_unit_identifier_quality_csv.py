from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_identifier_quality_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_identifier_quality_csv_detects_common_scalar_and_list_fields():
    text = export_units_to_identifier_quality_csv(
        [
            {
                "id": "a",
                "source_project": "zotero",
                "source_entity_type": "paper",
                "doi": "10.1/a",
                "metadata": {"isbn": ["978-1", "978-2"], "external_ids": {"pmid": "123"}},
            }
        ]
    )

    assert rows(text)[0] == {
        "unit_id": "a",
        "source": "zotero",
        "entity_type": "paper",
        "identifier_count": "4",
        "identifier_types": "doi; isbn; pmid",
        "missing_canonical_id": "false",
        "duplicate_identifier_values": "",
        "quality_flags": "",
    }


def test_identifier_quality_csv_reports_duplicates_deterministically():
    result = rows(export_units_to_identifier_quality_csv([{"id": "b", "url": "https://example.com", "metadata": {"external_ids": {"canonical": "https://example.com"}}}]))[0]

    assert result["duplicate_identifier_values"] == "https://example.com"
    assert result["quality_flags"] == "duplicate_identifier_values"


def test_identifier_quality_csv_flags_units_with_no_identifiers():
    result = rows(export_units_to_identifier_quality_csv([{"title": "No identifiers"}]))[0]

    assert result["identifier_count"] == "0"
    assert result["missing_canonical_id"] == "true"
    assert result["quality_flags"] == "no_identifiers; missing_canonical_id"
