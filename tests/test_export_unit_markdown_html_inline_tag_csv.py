from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_inline_tags_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_html_inline_tag_csv_exports_inline_tags_and_ignores_comments_blocks_fences():
    text = export_unit_markdown_html_inline_tags_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "Use <span class=\"x\">text</span> and <kbd>K</kbd>\n<div>\n<!-- <mark>x</mark> -->\n```\n<sup>x</sup>\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "1", "tag_name": "span", "closing": "false", "raw_tag": "<span class=\"x\">"},
        {"unit_id": "u", "title": "Unit", "line_number": "1", "tag_name": "span", "closing": "true", "raw_tag": "</span>"},
        {"unit_id": "u", "title": "Unit", "line_number": "1", "tag_name": "kbd", "closing": "false", "raw_tag": "<kbd>"},
        {"unit_id": "u", "title": "Unit", "line_number": "1", "tag_name": "kbd", "closing": "true", "raw_tag": "</kbd>"},
    ]
