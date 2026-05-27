from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_toc_entry_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_markdown_toc_entry_csv_reports_list_fragment_links_only():
    rows = _rows(export_unit_markdown_toc_entry_csv([{"id": "u1", "title": "TOC", "content": "- [Intro](#intro)\n  1. [Deep](#deep-section)\nInline [Nope](#nope)\n- [External](https://example.com)"}]))

    assert rows == [
        {"unit_id": "u1", "title": "TOC", "entry_text": "Intro", "fragment": "intro", "list_marker": "-", "indent": "0", "line_number": "1"},
        {"unit_id": "u1", "title": "TOC", "entry_text": "Deep", "fragment": "deep-section", "list_marker": "1.", "indent": "2", "line_number": "2"},
    ]
