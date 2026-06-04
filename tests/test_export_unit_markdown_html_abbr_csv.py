from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_html_abbr_csv import export_unit_markdown_html_abbr_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_abbr_csv_returns_stable_header_for_empty_input():
    assert export_unit_markdown_html_abbr_csv([]) == "unit_id,title,line_number,text,title_attribute,raw_html\n"


def test_abbr_csv_exports_text_title_line_unit_title_and_raw_html():
    content = 'Use <abbr title="Hypertext Markup Language">HTML</abbr> and <abbr title=\'Cascading Style Sheets\'><span>CSS</span></abbr>.'

    result = rows(export_unit_markdown_html_abbr_csv([{"id": "u", "title": "Glossary", "content": content}]))

    assert result == [
        {
            "unit_id": "u",
            "title": "Glossary",
            "line_number": "1",
            "text": "CSS",
            "title_attribute": "Cascading Style Sheets",
            "raw_html": "<abbr title='Cascading Style Sheets'><span>CSS</span></abbr>",
        },
        {
            "unit_id": "u",
            "title": "Glossary",
            "line_number": "1",
            "text": "HTML",
            "title_attribute": "Hypertext Markup Language",
            "raw_html": '<abbr title="Hypertext Markup Language">HTML</abbr>',
        },
    ]


def test_abbr_csv_ignores_fenced_code_blocks_and_decodes_entities():
    content = "```html\n<abbr title=\"skip\">NO</abbr>\n```\nRead <abbr title=\"Research &amp; Development\">R&amp;D</abbr>"

    result = rows(export_unit_markdown_html_abbr_csv([{"id": "u", "metadata": {"title": "Meta"}, "content": content}]))

    assert result == [
        {
            "unit_id": "u",
            "title": "Meta",
            "line_number": "4",
            "text": "R&D",
            "title_attribute": "Research & Development",
            "raw_html": '<abbr title="Research &amp; Development">R&amp;D</abbr>',
        }
    ]


def test_abbr_csv_path_mode_writes_identical_csv_and_returns_stats(tmp_path):
    output = tmp_path / "abbr.csv"
    units = [{"id": "u", "title": "T", "content": '<abbr title="Application Programming Interface">API</abbr>'}]
    expected = export_unit_markdown_html_abbr_csv(units)

    result = export_unit_markdown_html_abbr_csv(units, output)

    assert output.read_text(encoding="utf-8") == expected
    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
