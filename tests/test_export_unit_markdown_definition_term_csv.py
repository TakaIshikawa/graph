from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_definition_term_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_definition_term_csv_exports_pairs_and_continuations():
    text = export_unit_markdown_definition_term_csv(
        [{"id": "u", "title": "T", "source": "s", "content": "Term\n: First line\n  continued\n\nNext\n: Second"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "T", "source": "s", "term_line_number": "1", "definition_line_number": "2", "term_text": "Term", "definition_preview": "First line continued"},
        {"unit_id": "u", "title": "T", "source": "s", "term_line_number": "5", "definition_line_number": "6", "term_text": "Next", "definition_preview": "Second"},
    ]


def test_definition_term_csv_ignores_fenced_examples():
    text = export_unit_markdown_definition_term_csv([{"id": "u", "content": "```\nTerm\n: Hidden\n```\nVisible\n: Shown"}])

    assert _rows(text) == [{"unit_id": "u", "title": "", "source": "", "term_line_number": "5", "definition_line_number": "6", "term_text": "Visible", "definition_preview": "Shown"}]
