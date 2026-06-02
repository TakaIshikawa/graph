from __future__ import annotations

import csv
from io import StringIO

from graph.export.collection_orphan_unit_csv import export_collection_orphan_unit_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_orphan_unit_csv_reports_missing_and_dangling_collections():
    text = export_collection_orphan_unit_csv(
        [{"id": "c1"}],
        [
            {"id": "u1", "collection_id": "c1", "source_id": "s1"},
            {"id": "u2", "collection_id": "missing", "source_id": "s2"},
            {"id": "u3", "metadata": {"source_id": "s3"}},
        ],
    )

    assert rows(text) == [
        {"unit_id": "u2", "claimed_collection_id": "missing", "orphan_reason": "dangling_collection", "source_id": "s2"},
        {"unit_id": "u3", "claimed_collection_id": "", "orphan_reason": "missing_collection", "source_id": "s3"},
    ]
