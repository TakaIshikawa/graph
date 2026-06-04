from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_blockquote_cite_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_blockquote_cite_csv_exports_cited_multiline_quotes():
    text = export_units_to_markdown_html_blockquote_cite_csv(
        [
            {
                "id": "u",
                "title": "Quotes",
                "source_path": "quotes.md",
                "content": '<blockquote cite="https://example.org/source">\n<p>Quoted text</p>\n</blockquote>\n<blockquote>No cite</blockquote>',
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Quotes",
            "source_path": "quotes.md",
            "source": "",
            "line_number": "1",
            "cite_url": "https://example.org/source",
            "domain": "example.org",
            "text_preview": "Quoted text",
            "multiline": "true",
        }
    ]


def test_blockquote_cite_csv_ignores_fenced_html():
    text = export_units_to_markdown_html_blockquote_cite_csv(
        [{"id": "u", "content": '```\n<blockquote cite="https://skip.example">x</blockquote>\n```\n<blockquote cite=/local>ok</blockquote>'}]
    )

    rows = _rows(text)
    assert len(rows) == 1
    assert rows[0]["cite_url"] == "/local"
    assert rows[0]["line_number"] == "4"
