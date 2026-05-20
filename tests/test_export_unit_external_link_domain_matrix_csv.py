from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_external_link_domain_matrix_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_external_link_domain_matrix_csv_empty_input_returns_header():
    assert export_unit_external_link_domain_matrix_csv([]) == (
        "source_project,source_entity_type,domain,unit_count,linked_unit_count,link_count,sample_unit_ids\n"
    )


def test_export_unit_external_link_domain_matrix_csv_extracts_content_and_metadata_urls_deterministically():
    text = export_unit_external_link_domain_matrix_csv(
        [
            {"id": "u2", "source_project": "proj", "source_entity_type": "note", "content": "See https://EXAMPLE.com/a).", "metadata": {"links": ["http://docs.example.org/x", "mailto:nope@example.org"]}},
            {"id": "u1", "source_project": "proj", "source_entity_type": "note", "content": "Twice https://example.com/b https://example.com/c", "metadata": {"nested": {"url": "ftp://ignored"}}},
            {"id": "u3", "source_project": "proj", "source_entity_type": "task", "content": "none", "metadata": {}},
        ],
        sample_limit=1,
    )

    data = rows(text)
    assert data == [
        {"source_project": "proj", "source_entity_type": "note", "domain": "docs.example.org", "unit_count": "2", "linked_unit_count": "1", "link_count": "1", "sample_unit_ids": "u2"},
        {"source_project": "proj", "source_entity_type": "note", "domain": "example.com", "unit_count": "2", "linked_unit_count": "2", "link_count": "3", "sample_unit_ids": "u1"},
    ]


def test_export_unit_external_link_domain_matrix_csv_path_mode_writes_identical_content(tmp_path):
    units = [{"id": "u1", "source_project": "p", "source_entity_type": "n", "content": "https://example.com", "metadata": {}}]
    path = tmp_path / "domains.csv"

    expected = export_unit_external_link_domain_matrix_csv(units)
    stats = export_unit_external_link_domain_matrix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["sample_limit"] == 3
    assert stats["bytes_written"] == path.stat().st_size


def test_export_unit_external_link_domain_matrix_csv_validates_sample_limit():
    with pytest.raises(ValueError, match="sample_limit"):
        export_unit_external_link_domain_matrix_csv([], sample_limit=0)
