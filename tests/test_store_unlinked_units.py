from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(
    unit_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=unit_id,
        content=unit_id,
        content_type=content_type,
        tags=tags or [],
        created_at=f"2026-01-0{unit_id[-1]}T00:00:00+00:00",
    )


def test_get_unlinked_units_excludes_incoming_and_outgoing_edges(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        for unit_id in ("unit-1", "unit-2", "unit-3", "unit-4"):
            store.insert_unit(_unit(unit_id))
        store.insert_edge(
            KnowledgeEdge(
                id="edge-1",
                from_unit_id="unit-1",
                to_unit_id="unit-2",
                relation=EdgeRelation.RELATES_TO,
                source=EdgeSource.MANUAL,
            )
        )

        assert [unit.id for unit in store.get_unlinked_units()] == ["unit-4", "unit-3"]
    finally:
        store.close()


def test_get_unlinked_units_supports_filters_and_limit(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(_unit("unit-1", source_project=SourceProject.MAX, tags=["keep"]))
        store.insert_unit(
            _unit(
                "unit-2",
                source_project=SourceProject.FORTY_TWO,
                content_type=ContentType.METADATA,
                tags=["keep"],
            )
        )
        store.insert_unit(
            _unit(
                "unit-3",
                source_project=SourceProject.FORTY_TWO,
                content_type=ContentType.METADATA,
                tags=["keep", "later"],
            )
        )

        rows = store.get_unlinked_units(
            source_project="forty_two",
            content_type="metadata",
            tag="keep",
            limit=1,
        )

        assert [unit.id for unit in rows] == ["unit-3"]
    finally:
        store.close()


def test_get_unlinked_units_validates_limit(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError, match="limit"):
            store.get_unlinked_units(limit=-1)
    finally:
        store.close()
