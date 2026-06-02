from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_link_attribute_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_each_link_attribute():
    content = '[Docs](https://example.test){#main .primary download key="value"}'

    result = rows(export_units_to_markdown_link_attribute_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert [(row["attribute_name"], row["attribute_value"], row["attribute_kind"]) for row in result] == [
        ("class", "primary", "class"),
        ("download", "", "boolean"),
        ("id", "main", "id"),
        ("key", "value", "key_value"),
    ]
    assert result[0]["link_text"] == "Docs"
    assert result[0]["href"] == "https://example.test"


def test_ignores_reference_links_and_code_spans():
    content = "`[x](y){#code}` [ref][x]{#no} [ok](/ok){.yes}"

    result = rows(export_units_to_markdown_link_attribute_csv([{"id": "u", "content": content}]))

    assert [(row["link_text"], row["attribute_name"], row["attribute_value"]) for row in result] == [("ok", "class", "yes")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "link_attrs.csv"

    result = export_units_to_markdown_link_attribute_csv([{"id": "u", "content": "[x](y){flag}"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
