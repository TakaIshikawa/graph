from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_hashtag_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_hashtag_inventory_extracts_inline_tags_and_counts_by_line():
    text = export_units_to_hashtag_inventory_csv(
        [{"id": "a", "title": "Alpha", "content": "# Heading\nInline #Tag #tag\n```\n#skip\ncode #skip\n```\nAgain #tag"}]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "hashtag": "#Tag", "normalized_hashtag": "#tag", "line_number": "2", "occurrence_count": "1"},
        {"unit_id": "a", "title": "Alpha", "hashtag": "#tag", "normalized_hashtag": "#tag", "line_number": "2", "occurrence_count": "1"},
        {"unit_id": "a", "title": "Alpha", "hashtag": "#tag", "normalized_hashtag": "#tag", "line_number": "7", "occurrence_count": "1"},
    ]


def test_unit_hashtag_inventory_empty_input_returns_header():
    assert export_units_to_hashtag_inventory_csv([]) == "unit_id,title,hashtag,normalized_hashtag,line_number,occurrence_count\n"
