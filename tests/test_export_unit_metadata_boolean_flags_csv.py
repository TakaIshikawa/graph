from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_metadata_boolean_flags_csv import export_unit_metadata_boolean_flags_csv


def test_export_unit_metadata_boolean_flags_csv_traverses_and_normalizes_boolean_like_values():
    rows = list(csv.DictReader(StringIO(export_unit_metadata_boolean_flags_csv([{"id": "u1", "metadata": {"published": True, "status": "done", "items": [{"archived": "no"}], "plain": "hello"}}]))))

    assert [(row["metadata_key"], row["raw_value"], row["normalized_value"]) for row in rows] == [
        ("metadata.items.0.archived", "no", "false"),
        ("metadata.published", "True", "true"),
        ("metadata.status", "done", "true"),
    ]
