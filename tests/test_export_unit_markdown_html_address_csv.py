from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_address_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_address_csv_exports_contact_counts_and_skips_fences(tmp_path):
    content = """```
<address id="skip"><a href="mailto:skip@example.test">Skip</a></address>
```
<address id="office" class="vcard">
  Alice &amp; Bob
  <a href="mailto:team@example.test">Email</a>
  <a href='tel:+15550100'>Call</a>
  <a href="/map">Map</a>
</address>"""
    units = [{"id": "u", "title": "Contacts", "source_path": "contact.md", "content": content}]

    text = export_units_to_markdown_html_address_csv(units)
    result = _rows(text)

    assert len(result) == 1
    assert result[0]["id"] == "office"
    assert result[0]["class"] == "vcard"
    assert result[0]["text_preview"] == "Alice & Bob Email Call Map"
    assert result[0]["link_count"] == "3"
    assert result[0]["email_count"] == "1"
    assert result[0]["tel_count"] == "1"
    assert result[0]["has_nested_html"] == "true"

    output = tmp_path / "address.csv"
    stats = export_units_to_markdown_html_address_csv(units, output)
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert output.read_text() == text
