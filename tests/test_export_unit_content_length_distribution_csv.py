from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_units_to_content_length_distribution_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_content_length_distribution_default_buckets_and_averages():
    result = {row["bucket"]: row for row in rows(export_units_to_content_length_distribution_csv([
        {"id": "e", "content": ""},
        {"id": "s", "content": "abc"},
        {"id": "m", "content": "x" * 600},
    ]))}

    assert result["empty"]["unit_ids"] == "e"
    assert result["short"]["average_length"] == "3.00"
    assert result["medium"]["max_length"] == "600"


def test_content_length_distribution_custom_buckets_are_validated():
    text = export_units_to_content_length_distribution_csv([{"id": "u", "content": "abcd"}], buckets=[("tiny", 3), ("rest", None)])
    assert rows(text)[1]["unit_ids"] == "u"
    with pytest.raises(ValueError):
        export_units_to_content_length_distribution_csv([], buckets=[("bad", 1)])
