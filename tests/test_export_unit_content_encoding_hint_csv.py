from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_content_encoding_hint_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_content_encoding_hint_distinguishes_multilingual_text_from_mojibake():
    result = rows(export_units_to_content_encoding_hint_csv([{"id": "clean", "content": "こんにちは"}, {"id": "bad", "content": "FranÃ§ais �\x01"}]))
    parsed = {row["unit_id"]: row for row in result}

    assert parsed["clean"]["likely_mojibake"] == "false"
    assert parsed["bad"]["replacement_char_count"] == "1"
    assert parsed["bad"]["control_char_count"] == "1"
    assert parsed["bad"]["likely_mojibake"] == "true"
    assert "mojibake_markers" in parsed["bad"]["encoding_note"]


def test_content_encoding_hint_writes_empty_content_row(tmp_path):
    output = tmp_path / "encoding.csv"
    result = export_units_to_content_encoding_hint_csv([{"id": "u"}], output)

    assert result["rows_exported"] == 1
    assert rows(output.read_text(encoding="utf-8"))[0]["content_length"] == "0"
