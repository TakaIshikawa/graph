from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_duplicate_title_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_duplicate_title_groups_by_normalized_title_per_collection():
    result = rows(export_collection_duplicate_title_csv([
        {"id": "u1", "title": "  Same Title ", "source_id": "s1", "metadata": {"collections": ["c1", "c2"]}},
        {"id": "u2", "title": "same title", "source_id": "s2", "metadata": {"collection": "c1"}},
        {"id": "u3", "title": "Other", "metadata": {"collection": "c1"}},
    ]))

    assert len(result) == 1
    assert result[0]["collection"] == "c1"
    assert result[0]["duplicate_count"] == "2"
    assert result[0]["unit_ids"] == "u1; u2"
