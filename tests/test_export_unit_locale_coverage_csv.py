from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_locale_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_locale_coverage_csv_preserves_values_and_normalizes_bucket():
    text = export_unit_locale_coverage_csv(
        [
            {"id": "u2", "metadata": {"locale": "EN_US", "timezone": "America/New_York", "currency": "USD"}},
            {"id": "u1", "metadata": {"language": "ja", "country": "JP"}},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "u1",
            "locale": "",
            "language": "ja",
            "country": "JP",
            "region": "",
            "timezone": "",
            "currency": "",
            "locale_bucket": "ja-jp",
            "missing_locale_fields": "locale; region; timezone; currency",
        },
        {
            "unit_id": "u2",
            "locale": "EN_US",
            "language": "",
            "country": "",
            "region": "",
            "timezone": "America/New_York",
            "currency": "USD",
            "locale_bucket": "en-us",
            "missing_locale_fields": "language; country; region",
        },
    ]


def test_export_unit_locale_coverage_csv_path_mode(tmp_path):
    path = tmp_path / "locale.csv"
    stats = export_unit_locale_coverage_csv([{"id": "u1"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["locale_bucket"] == "unknown"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
