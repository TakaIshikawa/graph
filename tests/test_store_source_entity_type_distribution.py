from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    source_project: SourceProject | str,
    source_entity_type: str,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
    )


def test_source_entity_type_distribution_empty_store_returns_empty(store: Store):
    assert store.source_entity_type_distribution() == []


def test_source_entity_type_distribution_groups_and_orders_deterministically(store: Store):
    store.insert_unit(_unit("m-1", SourceProject.MAX, "note"))
    store.insert_unit(_unit("m-2", SourceProject.MAX, "note"))
    store.insert_unit(_unit("m-3", SourceProject.MAX, "task"))
    store.insert_unit(_unit("p-1", SourceProject.PRESENCE, "note"))
    store.insert_unit(_unit("p-2", SourceProject.PRESENCE, "entry"))

    assert store.source_entity_type_distribution() == [
        {"source_project": "max", "source_entity_type": "note", "count": 2},
        {"source_project": "max", "source_entity_type": "task", "count": 1},
        {"source_project": "presence", "source_entity_type": "entry", "count": 1},
        {"source_project": "presence", "source_entity_type": "note", "count": 1},
    ]


def test_source_entity_type_distribution_filters_independently_and_together(store: Store):
    store.insert_unit(_unit("m-1", SourceProject.MAX, "note"))
    store.insert_unit(_unit("m-2", SourceProject.MAX, "note", ContentType.FINDING))
    store.insert_unit(_unit("m-3", SourceProject.MAX, "task", ContentType.FINDING))
    store.insert_unit(_unit("p-1", SourceProject.PRESENCE, "note", ContentType.FINDING))

    assert store.source_entity_type_distribution(source_project=SourceProject.MAX) == [
        {"source_project": "max", "source_entity_type": "note", "count": 2},
        {"source_project": "max", "source_entity_type": "task", "count": 1},
    ]
    assert store.source_entity_type_distribution(content_type=ContentType.FINDING) == [
        {"source_project": "max", "source_entity_type": "note", "count": 1},
        {"source_project": "max", "source_entity_type": "task", "count": 1},
        {"source_project": "presence", "source_entity_type": "note", "count": 1},
    ]
    assert store.source_entity_type_distribution(
        source_project="max",
        content_type=ContentType.FINDING,
        min_count=2,
    ) == []


def test_source_entity_type_distribution_applies_min_count(store: Store):
    store.insert_unit(_unit("a", "alpha", "note"))
    store.insert_unit(_unit("b", "alpha", "note"))
    store.insert_unit(_unit("c", "alpha", "task"))

    assert store.source_entity_type_distribution(min_count=2) == [
        {"source_project": "alpha", "source_entity_type": "note", "count": 2},
    ]


@pytest.mark.parametrize("min_count", [0, -1, True])
def test_source_entity_type_distribution_validates_min_count(store: Store, min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        store.source_entity_type_distribution(min_count=min_count)
