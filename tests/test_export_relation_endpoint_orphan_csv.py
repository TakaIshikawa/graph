from __future__ import annotations

import csv
from io import StringIO

from graph.export.relation_endpoint_orphan_csv import export_relation_endpoint_orphan_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_endpoint_orphan_csv_emits_only_missing_endpoints():
    text = export_relation_endpoint_orphan_csv(
        [{"id": "u1"}, {"id": "u2"}],
        [
            {"id": "r0", "source_id": "u1", "target_id": "u2", "relation": "ok"},
            {"id": "r1", "source_id": "u1", "target_id": "missing", "relation": "ref"},
            {"id": "r2", "source_id": "absent", "target_id": "u2", "relation": "ref"},
        ],
    )

    assert rows(text) == [
        {"relation_id": "r1", "source_id": "u1", "target_id": "missing", "missing_source": "false", "missing_target": "true", "relation_type": "ref"},
        {"relation_id": "r2", "source_id": "absent", "target_id": "u2", "missing_source": "true", "missing_target": "false", "relation_type": "ref"},
    ]
