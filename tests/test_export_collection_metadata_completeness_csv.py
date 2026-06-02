from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_metadata_completeness_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_complete_collection():
    [row] = rows(export_collection_metadata_completeness_csv([{"id": "c", "title": "T", "metadata": {"description": "D", "tags": ["x"]}}], required_keys=("title", "description", "tags")))
    assert row["present_key_count"] == "3"
    assert row["present_keys"] == "title; description; tags"
    assert row["completeness_score"] == "1.00"


def test_missing_and_empty_values():
    [row] = rows(export_collection_metadata_completeness_csv([{"id": "c", "title": "", "metadata": {"description": None, "tags": []}}], required_keys=("title", "description", "tags", "source")))
    assert row["empty_keys"] == "title; description; tags; source"
    assert row["missing_keys"] == "source"


def test_list_valued_metadata_counts_present():
    [row] = rows(export_collection_metadata_completeness_csv([{"id": "c", "metadata": {"tags": ["a"]}}], required_keys=("tags",)))
    assert row["present_key_count"] == "1"


def test_deterministic_order():
    text = export_collection_metadata_completeness_csv([{"id": "b"}, {"id": "a"}], required_keys=("title",))
    assert [row["collection_id"] for row in rows(text)] == ["a", "b"]
