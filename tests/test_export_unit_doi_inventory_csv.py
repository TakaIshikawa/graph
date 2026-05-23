from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_doi_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_doi_inventory_empty_input_has_header():
    assert (
        export_unit_doi_inventory_csv([]) == "unit_id,title,doi,doi_source,normalized_doi,has_doi\n"
    )


def test_unit_doi_inventory_detects_metadata_content_and_url_dois():
    text = export_unit_doi_inventory_csv(
        [
            {"id": "m", "title": "Meta", "metadata": {"doi": "https://doi.org/10.1000/ABC.Def"}},
            {"id": "c", "title": "Content", "content": "See doi:10.5555/Test-1.", "metadata": {}},
            {"id": "u", "title": "URL", "metadata": {"url": "https://doi.org/10.2000/XYZ"}},
            {"id": "n", "title": "None", "metadata": {}},
        ]
    )

    result = {row["unit_id"]: row for row in rows(text)}
    assert result["m"]["normalized_doi"] == "10.1000/abc.def"
    assert result["m"]["doi_source"] == "metadata.doi"
    assert result["c"]["normalized_doi"] == "10.5555/test-1"
    assert result["u"]["doi_source"] == "url"
    assert result["n"]["has_doi"] == "false"


def test_unit_doi_inventory_path_mode(tmp_path):
    path = tmp_path / "doi.csv"
    stats = export_unit_doi_inventory_csv(
        [{"unit_id": "u1", "title": "One", "metadata": {"doi": "10.1001/ABC"}}], path
    )

    assert rows(path.read_text(encoding="utf-8"))[0]["has_doi"] == "true"
    assert stats["rows_exported"] == 1
