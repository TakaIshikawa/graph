from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_quote_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_quote_inventory_groups_blockquotes_metadata_and_attribution():
    text = "> one\n> two\n\n> nested\n> - Author"
    result = rows(export_units_to_quote_inventory_csv([{"id": "q", "content": text, "metadata": {"quotes": ["meta quote", "second"]}}]))[0]

    assert result["quote_line_count"] == "4"
    assert result["quote_block_count"] == "2"
    assert result["metadata_quote_count"] == "2"
    assert result["has_attribution_marker"] == "true"


def test_quote_inventory_missing_content_and_path_write(tmp_path):
    output = tmp_path / "quotes.csv"
    result = export_units_to_quote_inventory_csv([{"id": "q", "metadata": {"source": "Book"}}], output)

    assert result["bytes_written"] == output.stat().st_size
    row = rows(output.read_text(encoding="utf-8"))[0]
    assert row["quote_line_count"] == "0"
    assert row["has_attribution_marker"] == "true"
