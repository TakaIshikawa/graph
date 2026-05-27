from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_link_fragment_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_markdown_link_fragment_csv_exports_external_and_internal_fragments():
    text = export_unit_markdown_link_fragment_csv(
        [{"id": "u1", "title": "One", "content": "[Ext](https://example.com/doc#part-1)\n[Local](#heading)"}]
    )

    assert [(row["link_text"], row["fragment"], row["is_internal"]) for row in rows(text)] == [
        ("Ext", "part-1", "false"),
        ("Local", "heading", "true"),
    ]


def test_unit_markdown_link_fragment_csv_ignores_links_without_fragments():
    text = export_unit_markdown_link_fragment_csv([{"id": "u1", "title": "One", "content": "[Plain](https://example.com/doc)"}])

    assert rows(text) == []


def test_unit_markdown_link_fragment_csv_preserves_fragment_content_after_hash():
    text = export_unit_markdown_link_fragment_csv([{"id": "u1", "title": "One", "content": "[A](https://x.test/a#sec%202)"}])

    assert rows(text)[0]["fragment"] == "sec%202"
