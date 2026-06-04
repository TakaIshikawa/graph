from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_sup_sub_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_sup_sub_csv_exports_tag_name_text_and_reference_flag():
    text = export_units_to_markdown_html_sup_sub_csv(
        [{"id": "u", "metadata": {"source": "chem"}, "content": "Water H<sub>2</sub>O and note<sup>1</sup> plus x<sup>note</sup>."}]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "",
            "source_path": "",
            "source": "chem",
            "line_number": "1",
            "tag_name": "sub",
            "span_text": "2",
            "is_reference_like": "false",
            "context_preview": "Water H2O and note1 plus xnote.",
        },
        {
            "unit_id": "u",
            "title": "",
            "source_path": "",
            "source": "chem",
            "line_number": "1",
            "tag_name": "sup",
            "span_text": "1",
            "is_reference_like": "true",
            "context_preview": "Water H2O and note1 plus xnote.",
        },
        {
            "unit_id": "u",
            "title": "",
            "source_path": "",
            "source": "chem",
            "line_number": "1",
            "tag_name": "sup",
            "span_text": "note",
            "is_reference_like": "false",
            "context_preview": "Water H2O and note1 plus xnote.",
        },
    ]


def test_sup_sub_csv_ignores_fenced_spans():
    text = export_units_to_markdown_html_sup_sub_csv([{"id": "u", "content": "```\n<sup>1</sup>\n```\n<sub>x</sub>"}])

    assert [(row["tag_name"], row["line_number"]) for row in _rows(text)] == [("sub", "4")]
