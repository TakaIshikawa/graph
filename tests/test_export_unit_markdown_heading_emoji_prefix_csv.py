from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_heading_emoji_prefix_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_heading_emoji_prefix_csv_exports_atx_and_setext_headings():
    text = export_unit_markdown_heading_emoji_prefix_csv([{"id": "u", "title": "T", "source": "s", "content": "# 🚀 Launch\nPlain\n---\n✨ Spark\n==="}])

    assert _rows(text) == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "heading_depth": "1", "emoji": "🚀", "heading_text": "Launch"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "4", "heading_depth": "1", "emoji": "✨", "heading_text": "Spark"},
    ]


def test_heading_emoji_prefix_csv_ignores_non_emoji_and_fenced_headings():
    text = export_unit_markdown_heading_emoji_prefix_csv([{"id": "u", "content": "# Plain\n```\n# ✅ Hidden\n```\n## ✅ Visible"}])

    assert _rows(text) == [{"unit_id": "u", "title": "", "source": "", "line_number": "5", "heading_depth": "2", "emoji": "✅", "heading_text": "Visible"}]
