from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    source_project: SourceProject | str,
    source_id: str,
    source_entity_type: str,
    title: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type=source_entity_type,
        title=title,
        content=f"Content for {title}",
        content_type=ContentType.INSIGHT,
    )


def test_find_source_id_collisions_groups_orders_filters_and_limits(store: Store):
    store.insert_unit(_unit("max-c-3", SourceProject.MAX, "shared", "quote", "Max quote"))
    store.insert_unit(_unit("max-c-1", SourceProject.MAX, "shared", "note", "Max note"))
    store.insert_unit(_unit("max-c-2", SourceProject.MAX, "shared", "artifact", "Max artifact"))
    store.insert_unit(_unit("ft-c-2", SourceProject.FORTY_TWO, "alpha", "note", "FT note"))
    store.insert_unit(
        _unit("ft-c-1", SourceProject.FORTY_TWO, "alpha", "artifact", "FT artifact")
    )
    store.insert_unit(_unit("max-a-1", SourceProject.MAX, "alpha", "note", "Max alpha note"))
    store.insert_unit(
        _unit("max-a-2", SourceProject.MAX, "alpha", "artifact", "Max alpha artifact")
    )
    store.insert_unit(_unit("single", SourceProject.MAX, "unique", "note", "Unique"))
    store.insert_unit(_unit("blank-1", SourceProject.MAX, "", "note", "Blank note"))
    store.insert_unit(_unit("blank-2", SourceProject.MAX, "   ", "artifact", "Blank artifact"))

    assert store.find_source_id_collisions(limit=2) == [
        {
            "source_project": "max",
            "source_id": "shared",
            "count": 3,
            "unit_ids": ["max-c-1", "max-c-2", "max-c-3"],
            "titles": ["Max note", "Max artifact", "Max quote"],
        },
        {
            "source_project": "forty_two",
            "source_id": "alpha",
            "count": 2,
            "unit_ids": ["ft-c-1", "ft-c-2"],
            "titles": ["FT artifact", "FT note"],
        },
    ]
    assert store.find_source_id_collisions(source_project="max") == [
        {
            "source_project": "max",
            "source_id": "shared",
            "count": 3,
            "unit_ids": ["max-c-1", "max-c-2", "max-c-3"],
            "titles": ["Max note", "Max artifact", "Max quote"],
        },
        {
            "source_project": "max",
            "source_id": "alpha",
            "count": 2,
            "unit_ids": ["max-a-1", "max-a-2"],
            "titles": ["Max alpha note", "Max alpha artifact"],
        },
    ]
    assert store.find_source_id_collisions(limit=0) == []


def test_find_source_id_collisions_returns_empty_list_without_collisions(store: Store):
    store.insert_unit(_unit("unit-a", SourceProject.MAX, "a", "note", "A"))
    store.insert_unit(_unit("unit-b", SourceProject.FORTY_TWO, "a", "note", "B"))
    store.insert_unit(_unit("unit-c", SourceProject.MAX, "", "artifact", "Blank"))

    assert store.find_source_id_collisions() == []


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_find_source_id_collisions_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        store.find_source_id_collisions(limit=limit)
