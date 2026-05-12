from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_metadata_key_matrix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, metadata: dict):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
        created_at=TIME,
        ingested_at=TIME,
        updated_at=TIME,
    )


def rows(text: str):
    return list(csv.DictReader(StringIO(text)))


def test_metadata_key_matrix_discovers_sorted_presence_columns():
    text = export_metadata_key_matrix_csv(
        [
            unit("b", "Beta", {"priority": 2, "status": "done"}),
            unit("a", "Alpha", {"owner": "Ada", "priority": 1}),
        ]
    )

    assert text.splitlines()[0] == "unit_id,title,source,type,owner,priority,status"
    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "max",
            "type": "note",
            "owner": "1",
            "priority": "1",
            "status": "",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "max",
            "type": "note",
            "owner": "",
            "priority": "1",
            "status": "1",
        },
    ]


def test_metadata_key_matrix_supports_explicit_key_order_and_values(tmp_path):
    path = tmp_path / "metadata.csv"
    stats = export_metadata_key_matrix_csv(
        [unit("a", "Alpha", {"nested": {"b": 2, "a": 1}, "tags": ["z", "a"], "active": True})],
        path,
        keys=["tags", "nested", "active", "missing"],
        include_values=True,
    )
    row = rows(path.read_text(encoding="utf-8"))[0]

    assert list(row) == ["unit_id", "title", "source", "type", "tags", "nested", "active", "missing"]
    assert row["tags"] == '["z","a"]'
    assert row["nested"] == '{"a":1,"b":2}'
    assert row["active"] == "true"
    assert row["missing"] == ""
    assert stats["metadata_key_count"] == 4


def test_metadata_key_matrix_is_deterministic():
    units = [
        unit("b", "Beta", {"priority": 2, "status": "done"}),
        unit("a", "Alpha", {"owner": "Ada", "priority": 1}),
    ]

    assert export_metadata_key_matrix_csv(units) == export_metadata_key_matrix_csv(reversed(units))
