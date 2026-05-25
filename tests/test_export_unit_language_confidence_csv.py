from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_language_confidence_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_language_confidence_reads_common_fields_and_flags_mismatch():
    result = {row["unit_id"]: row for row in rows(export_units_to_language_confidence_csv([
        {"id": "a", "metadata": {"language": "en-US", "detected_language": "en", "language_confidence": 0.92}},
        {"id": "b", "metadata": {"lang": "ja", "detected_language": "en"}},
    ]))}

    assert result["a"]["confidence"] == "0.92"
    assert result["a"]["mismatch_flag"] == "false"
    assert result["b"]["mismatch_flag"] == "true"
    assert result["b"]["declared_language"] == "ja"
