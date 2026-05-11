from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(unit_id: str, metadata: dict, source_project=SourceProject.MAX) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=unit_id,
        content=unit_id,
        metadata=metadata,
    )


def test_metadata_schema_conflicts_reports_paths_with_multiple_types(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(_unit("a", {"rating": 5, "nested": {"score": "high"}}))
        store.insert_unit(_unit("b", {"rating": "5", "nested": {"score": 10}}))
        store.insert_unit(_unit("c", {"rating": 6, "tags": ["one"]}, SourceProject.FORTY_TWO))

        rows = store.metadata_schema_conflicts()

        assert rows[0]["path"] == "rating"
        assert rows[0]["count"] == 3
        assert rows[0]["value_types"] == ["integer", "string"]
        assert set(rows[0]["source_projects"]) == {"max", "forty_two"}
        assert {row["path"] for row in rows} >= {"rating", "nested.score"}
    finally:
        store.close()


def test_metadata_schema_conflicts_supports_prefix_min_type_count_and_limit(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(_unit("a", {"metrics": {"score": 1, "flag": True}}))
        store.insert_unit(_unit("b", {"metrics": {"score": "1", "flag": False}}))
        store.insert_unit(_unit("c", {"metrics": {"score": 1.5, "flag": True}}))

        rows = store.metadata_schema_conflicts(prefix="metrics", min_type_count=3, limit=1)

        assert [row["path"] for row in rows] == ["metrics.score"]
        assert rows[0]["value_types"] == ["integer", "number", "string"]
    finally:
        store.close()


def test_metadata_schema_conflicts_validates_arguments(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError, match="min_type_count"):
            store.metadata_schema_conflicts(min_type_count=0)
        with pytest.raises(ValueError, match="limit"):
            store.metadata_schema_conflicts(limit=0)
    finally:
        store.close()
