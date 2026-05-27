from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_bare_url_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_bare_urls_with_line_numbers():
    found = rows(export_units_to_bare_url_csv([{"id": "u", "content": "See https://Example.com/a\nand http://b.test"}]))
    assert [row["line_number"] for row in found] == ["1", "2"]
    assert found[0]["domain"] == "example.com"


def test_markdown_links_and_autolinks_excluded():
    assert rows(export_units_to_bare_url_csv([{"id": "u", "content": "[x](https://a.test)\n<https://b.test>"}])) == []


def test_punctuation_trimming():
    [row] = rows(export_units_to_bare_url_csv([{"id": "u", "content": "Go https://a.test/path)."}]))
    assert row["url"] == "https://a.test/path"


def test_multiple_urls_on_one_line():
    found = rows(export_units_to_bare_url_csv([{"id": "u", "content": "https://a.test https://b.test"}]))
    assert [row["domain"] for row in found] == ["a.test", "b.test"]
