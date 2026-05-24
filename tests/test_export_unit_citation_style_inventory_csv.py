from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_citation_style_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_citation_style_inventory_detects_content_and_metadata_styles():
    text = export_unit_citation_style_inventory_csv(
        [
            {
                "id": "u1",
                "content": "Prior work (Smith, 2020) supports this [12]. See DOI:10.1000/xyz.",
                "metadata": {"references": ["https://example.test/ref"], "citation": "[^1]"},
            }
        ]
    )

    row = rows(text)[0]
    assert row["unit_id"] == "u1"
    assert row["detected_styles"] == "apa_parenthetical; numeric_bracket; doi; footnote; url_reference"
    assert row["has_doi"] == "true"
    assert row["has_url_reference"] == "true"
    assert int(row["citation_count"]) >= 5


def test_unit_citation_style_inventory_writes_path_metadata(tmp_path):
    path = tmp_path / "citations.csv"
    stats = export_unit_citation_style_inventory_csv([{"id": "u1", "content": "No citations"}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
