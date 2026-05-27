from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_wikilink_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_wikilinks_with_target_section_and_alias():
    result = rows(export_units_to_markdown_wikilink_csv([{"id": "u1", "title": "Note", "content": "See [[Target]] and [[Target#Heading|Alias]]."}]))
    assert result == [
        {"unit_id": "u1", "title": "Note", "target": "Target", "section": "", "alias": "", "line_number": "1", "context": "See [[Target]] and [[Target#Heading|Alias]]."},
        {"unit_id": "u1", "title": "Note", "target": "Target", "section": "Heading", "alias": "Alias", "line_number": "1", "context": "See [[Target]] and [[Target#Heading|Alias]]."},
    ]


def test_ignores_escaped_and_malformed_wikilinks():
    text = export_units_to_markdown_wikilink_csv([{"id": "u1", "content": r"\[[Escaped]] [[ ]] [[Target|Alias|Bad]] [[Broken"}])
    assert rows(text) == []


def test_importable_from_graph_export():
    assert callable(export_units_to_markdown_wikilink_csv)
