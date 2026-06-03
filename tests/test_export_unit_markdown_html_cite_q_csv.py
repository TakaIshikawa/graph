from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_cite_q_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_cite_q_csv_exports_cite_text_and_q_cite_urls():
    text = export_unit_markdown_html_cite_q_csv(
        [
            {
                "id": "u",
                "title": "Quotes",
                "content": '<cite> Example   Work </cite> and <q cite="https://example.test/source">quoted text</q>',
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Quotes",
            "line_number": "1",
            "tag": "cite",
            "text": "Example Work",
            "cite_url": "",
            "raw_html": "<cite> Example   Work </cite>",
        },
        {
            "unit_id": "u",
            "title": "Quotes",
            "line_number": "1",
            "tag": "q",
            "text": "quoted text",
            "cite_url": "https://example.test/source",
            "raw_html": '<q cite="https://example.test/source">quoted text</q>',
        },
    ]


def test_cite_q_csv_supports_case_insensitive_closing_tags_and_skips_fences():
    text = export_unit_markdown_html_cite_q_csv(
        [
            {"id": "b", "content": "```html\n<cite>Skip</cite>\n```\n<q cite='urn:x'>Keep <em>nested</em></Q>"},
            {"id": "a", "metadata": {"title": "Meta"}, "content": "<CITE>Alpha</cite>"},
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "a",
            "title": "Meta",
            "line_number": "1",
            "tag": "cite",
            "text": "Alpha",
            "cite_url": "",
            "raw_html": "<CITE>Alpha</cite>",
        },
        {
            "unit_id": "b",
            "title": "",
            "line_number": "4",
            "tag": "q",
            "text": "Keep nested",
            "cite_url": "urn:x",
            "raw_html": "<q cite='urn:x'>Keep <em>nested</em></Q>",
        },
    ]


def test_cite_q_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "cite-q.csv"
    units = [{"id": "u", "content": "<cite>One</cite>"}]

    expected = export_unit_markdown_html_cite_q_csv(units)
    stats = export_unit_markdown_html_cite_q_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
