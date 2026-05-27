from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_blockquote_attribution_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_blockquote_attribution_groups_blocks_and_detects_attribution():
    text = export_units_to_blockquote_attribution_csv(
        [{"id": "a", "title": "Alpha", "content": "> First\n> second\n> -- Ada\n\n> Other quote\nSource: Notes"}]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "quote_index": "1", "start_line": "1", "end_line": "3", "has_attribution": "true", "attribution_text": "Ada", "quote_preview": "First second -- Ada"},
        {"unit_id": "a", "title": "Alpha", "quote_index": "2", "start_line": "5", "end_line": "5", "has_attribution": "true", "attribution_text": "Notes", "quote_preview": "Other quote"},
    ]


def test_unit_blockquote_attribution_empty_input_returns_header():
    assert export_units_to_blockquote_attribution_csv([]) == "unit_id,title,quote_index,start_line,end_line,has_attribution,attribution_text,quote_preview\n"
