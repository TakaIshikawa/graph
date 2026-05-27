from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_tag_overlap_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_no_overlap():
    [row] = rows(export_collection_tag_overlap_csv([{"id": "a", "tags": ["x"]}, {"id": "b", "tags": ["y"]}]))
    assert row["shared_tag_count"] == "0"
    assert row["jaccard_similarity"] == "0.0000"


def test_partial_overlap_case_normalized():
    [row] = rows(export_collection_tag_overlap_csv([{"id": "a", "tags": ["AI", "ml"]}, {"id": "b", "tags": ["ai", "data"]}]))
    assert row["shared_tags"] == "AI"
    assert row["only_a_count"] == "1"


def test_identical_tags():
    [row] = rows(export_collection_tag_overlap_csv([{"id": "a", "tags": ["x"]}, {"id": "b", "tags": ["X"]}]))
    assert row["jaccard_similarity"] == "1.0000"


def test_stable_pair_ordering():
    text = export_collection_tag_overlap_csv([{"id": "c"}, {"id": "a"}, {"id": "b"}])
    assert [(row["collection_id_a"], row["collection_id_b"]) for row in rows(text)] == [("a", "b"), ("a", "c"), ("b", "c")]
