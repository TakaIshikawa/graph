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
    content_type: ContentType = ContentType.INSIGHT,
    source_entity_type: str = "note",
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


def test_source_content_type_mix_empty_store_returns_empty(store: Store):
    assert store.source_content_type_mix() == []


def test_source_content_type_mix_groups_and_orders_deterministically(store: Store):
    store.insert_unit(_unit("m-1", SourceProject.MAX, ContentType.INSIGHT))
    store.insert_unit(_unit("m-2", SourceProject.MAX, ContentType.INSIGHT))
    store.insert_unit(_unit("m-3", SourceProject.MAX, ContentType.FINDING))
    store.insert_unit(_unit("p-1", SourceProject.PRESENCE, ContentType.FINDING))
    store.insert_unit(_unit("p-2", SourceProject.PRESENCE, ContentType.ARTIFACT))

    assert store.source_content_type_mix() == [
        {"source_project": "max", "content_type": "insight", "count": 2},
        {"source_project": "max", "content_type": "finding", "count": 1},
        {"source_project": "presence", "content_type": "artifact", "count": 1},
        {"source_project": "presence", "content_type": "finding", "count": 1},
    ]


def test_source_content_type_mix_filters_independently_and_together(store: Store):
    store.insert_unit(_unit("m-1", SourceProject.MAX, ContentType.INSIGHT, "note"))
    store.insert_unit(_unit("m-2", SourceProject.MAX, ContentType.FINDING, "note"))
    store.insert_unit(_unit("m-3", SourceProject.MAX, ContentType.FINDING, "task"))
    store.insert_unit(_unit("p-1", SourceProject.PRESENCE, ContentType.FINDING, "note"))

    assert store.source_content_type_mix(source_project=SourceProject.MAX) == [
        {"source_project": "max", "content_type": "finding", "count": 2},
        {"source_project": "max", "content_type": "insight", "count": 1},
    ]
    assert store.source_content_type_mix(source_entity_type="note") == [
        {"source_project": "max", "content_type": "finding", "count": 1},
        {"source_project": "max", "content_type": "insight", "count": 1},
        {"source_project": "presence", "content_type": "finding", "count": 1},
    ]
    assert store.source_content_type_mix(
        source_project="max",
        source_entity_type="note",
        min_count=2,
    ) == []


def test_source_content_type_mix_applies_min_count(store: Store):
    store.insert_unit(_unit("a", "alpha", ContentType.INSIGHT))
    store.insert_unit(_unit("b", "alpha", ContentType.INSIGHT))
    store.insert_unit(_unit("c", "alpha", ContentType.FINDING))

    assert store.source_content_type_mix(min_count=2) == [
        {"source_project": "alpha", "content_type": "insight", "count": 2},
    ]


@pytest.mark.parametrize("min_count", [0, -1, True])
def test_source_content_type_mix_validates_min_count(store: Store, min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        store.source_content_type_mix(min_count=min_count)
