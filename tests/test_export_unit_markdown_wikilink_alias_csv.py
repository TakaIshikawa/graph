from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_wikilink_aliases_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_wikilink_alias_csv_splits_aliases_and_detects_heading_fragments():
    text = export_unit_markdown_wikilink_aliases_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "[[Page|Alias]] [[Page#Heading|Jump]] [[Plain]]\n```\n[[No|Nope]]\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "1", "target": "Page", "alias": "Alias", "has_heading_fragment": "false"},
        {"unit_id": "u", "title": "Unit", "line_number": "1", "target": "Page#Heading", "alias": "Jump", "has_heading_fragment": "true"},
    ]
