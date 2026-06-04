from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_abbr_title_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_abbr_title_csv_exports_expansion_and_context():
    text = export_units_to_markdown_html_abbr_title_csv(
        [{"id": "u", "path": "abbr.md", "content": 'Read the <abbr title="HyperText Markup Language">HTML</abbr> spec.'}]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "",
            "source_path": "abbr.md",
            "source": "",
            "line_number": "1",
            "abbr_text": "HTML",
            "title": "HyperText Markup Language",
            "context_preview": "Read the HTML spec.",
        }
    ]


def test_abbr_title_csv_skips_titleless_and_fenced_abbr():
    text = export_units_to_markdown_html_abbr_title_csv(
        [{"id": "u", "content": "```html\n<abbr title='Skip'>S</abbr>\n```\n<abbr>bad</abbr>\n<abbr title=Okay>OK</abbr>"}]
    )

    assert [row["abbr_text"] for row in _rows(text)] == ["OK"]
