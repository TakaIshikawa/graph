import csv
from io import StringIO

from graph.export import export_units_to_markdown_wikilink_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_wikilink_inventory_exports_plain_and_aliased_links():
    content = "See [[Target]] and [[Other Page|the alias]].\nEscaped \\[[Nope]]"

    assert rows(export_units_to_markdown_wikilink_inventory_csv([{"id": "u", "title": "T", "content": content}])) == [
        {"unit_id": "u", "title": "T", "target": "Target", "alias": "", "raw": "[[Target]]", "line": "1"},
        {"unit_id": "u", "title": "T", "target": "Other Page", "alias": "the alias", "raw": "[[Other Page|the alias]]", "line": "1"},
    ]


def test_wikilink_inventory_empty_content_emits_header_only():
    assert rows(export_units_to_markdown_wikilink_inventory_csv([{"id": "u"}])) == []


def test_wikilink_inventory_writes_path(tmp_path):
    path = tmp_path / "wikilinks.csv"
    stats = export_units_to_markdown_wikilink_inventory_csv([{"id": "u", "content": "[[A]]"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["target"] == "A"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] > 0
