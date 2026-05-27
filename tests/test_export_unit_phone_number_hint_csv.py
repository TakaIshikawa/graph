from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_phone_number_hint_csv import export_units_to_phone_number_hint_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_phone_number_hint_csv_header_only_for_empty_input():
    assert export_units_to_phone_number_hint_csv([]) == "unit_id,title,phone_hint,digit_count,source_field,context\n"


def test_export_units_to_phone_number_hint_csv_detects_common_numbers_and_skips_dates():
    rows = _rows(
        export_units_to_phone_number_hint_csv(
            [{"id": "u1", "title": "Call", "content": "Call (415) 555-1212, not 2026-05-27.", "metadata": {"contact": "+44 20 7946 0958"}}]
        )
    )

    assert [row["phone_hint"] for row in rows] == ["(415) 555-1212", "+44 20 7946 0958"]
    assert rows[0]["digit_count"] == "10"
    assert rows[1]["source_field"] == "metadata.contact"
