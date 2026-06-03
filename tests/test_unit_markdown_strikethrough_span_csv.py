from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_strikethrough_span_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_strikethrough_span_csv_exports_multiple_spans_in_document_order():
    text = export_units_to_markdown_strikethrough_span_csv([{"id": "u", "content": "Keep ~~first~~, then ~~second!~~."}])

    assert _rows(text) == [
        {"unit_id": "u", "text": "first", "line_number": "1", "column_number": "6", "character_count": "5"},
        {"unit_id": "u", "text": "second!", "line_number": "1", "column_number": "22", "character_count": "7"},
    ]


def test_strikethrough_span_csv_ignores_unmatched_code_spans_and_fences():
    text = export_units_to_markdown_strikethrough_span_csv(
        [{"id": "u", "content": "No ~~open only and `~~code~~`\n```md\n~~fenced~~\n```\nNow ~~real~~"}]
    )

    assert _rows(text) == [{"unit_id": "u", "text": "real", "line_number": "5", "column_number": "5", "character_count": "4"}]


def test_strikethrough_span_csv_returns_header_only_for_no_matches():
    text = export_units_to_markdown_strikethrough_span_csv([{"id": "u", "content": "plain text"}])

    assert _rows(text) == []


def test_strikethrough_span_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "strike-spans.csv"
    units = [{"unit_id": "u1", "content": "~~gone~~"}]

    expected = export_units_to_markdown_strikethrough_span_csv(units)
    stats = export_units_to_markdown_strikethrough_span_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
