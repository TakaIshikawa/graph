from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_microdata_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_markdown_html_microdata_elements_and_skips_fences(tmp_path):
    content = """```
<div itemscope itemtype="https://schema.org/Skip" itemprop="bad">Skip</div>
```
<div id="book" class="featured" itemscope itemtype="https://schema.org/Book" itemid="urn:isbn:1" itemref="author">The <span>Book</span></div>
<span itemprop="name">Example Book</span>
<a itemprop="url" href="https://example.test/book">Read</a>"""
    units = [{"id": "u", "title": "Unit", "source_path": "notes/u.md", "content": content}]

    text = export_units_to_markdown_html_microdata_csv(units)
    rows = _rows(text)

    assert len(rows) == 3
    assert rows[0]["unit_id"] == "u"
    assert rows[0]["title"] == "Unit"
    assert rows[0]["source_path"] == "notes/u.md"
    assert rows[0]["line_number"] == "4"
    assert rows[0]["tag_name"] == "div"
    assert rows[0]["itemscope"] == "true"
    assert rows[0]["itemtype"] == "https://schema.org/Book"
    assert rows[0]["itemid"] == "urn:isbn:1"
    assert rows[0]["itemref"] == "author"
    assert rows[0]["text_preview"] == "The Book"
    assert rows[0]["id"] == "book"
    assert rows[0]["class"] == "featured"
    assert rows[1]["tag_name"] == "span"
    assert rows[1]["itemprop"] == "name"
    assert rows[2]["tag_name"] == "a"
    assert rows[2]["itemprop"] == "url"

    output = tmp_path / "microdata.csv"
    result = export_units_to_markdown_html_microdata_csv(units, output)
    assert result["rows_exported"] == 3
    assert output.read_text() == text
