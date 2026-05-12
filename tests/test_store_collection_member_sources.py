from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
    )


def test_collection_member_sources_missing_collection_returns_clear_payload(store: Store):
    assert store.collection_member_sources("missing") == {
        "collection": "missing",
        "total_units": 0,
        "source_project_counts": {},
        "source_entity_type_counts": {},
        "rows": [],
        "error": "collection_not_found",
        "message": "Collection not found: missing",
    }


def test_collection_member_sources_empty_collection_when_included(store: Store):
    store.create_collection("empty", metadata={"owner": "research"})

    summary = store.collection_member_sources("empty", include_empty=True)

    assert summary["collection"]["name"] == "empty"
    assert summary["collection"]["metadata"] == {"owner": "research"}
    assert summary["collection"]["unit_count"] == 0
    assert summary["total_units"] == 0
    assert summary["source_project_counts"] == {}
    assert summary["source_entity_type_counts"] == {}
    assert summary["rows"] == []


def test_collection_member_sources_counts_members_only(store: Store):
    store.create_collection("review")
    max_note = store.insert_unit(_unit("max-note", source_project=SourceProject.MAX, source_entity_type="note"))
    max_task = store.insert_unit(_unit("max-task", source_project=SourceProject.MAX, source_entity_type="task"))
    me_note = store.insert_unit(_unit("me-note", source_project=SourceProject.ME, source_entity_type="note"))
    nonmember = store.insert_unit(
        _unit("presence-note", source_project=SourceProject.PRESENCE, source_entity_type="note")
    )
    for unit in (max_note, max_task, me_note):
        store.add_unit_to_collection("review", unit.id)

    summary = store.collection_member_sources("review")

    assert nonmember.id
    assert summary["total_units"] == 3
    assert summary["source_project_counts"] == {"max": 2, "me": 1}
    assert summary["source_entity_type_counts"] == {"note": 2, "task": 1}
    assert summary["rows"] == [
        {"source_project": "max", "source_entity_type": "note", "count": 1},
        {"source_project": "max", "source_entity_type": "task", "count": 1},
        {"source_project": "me", "source_entity_type": "note", "count": 1},
    ]


def test_collection_member_sources_applies_min_count(store: Store):
    store.create_collection("review")
    for unit_id in ("a", "b"):
        unit = store.insert_unit(_unit(unit_id, source_project="alpha", source_entity_type="note"))
        store.add_unit_to_collection("review", unit.id)
    task = store.insert_unit(_unit("c", source_project="alpha", source_entity_type="task"))
    store.add_unit_to_collection("review", task.id)

    summary = store.collection_member_sources("review", min_count=2)

    assert summary["total_units"] == 3
    assert summary["rows"] == [
        {"source_project": "alpha", "source_entity_type": "note", "count": 2},
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_count": 0}, "min_count must be a positive integer"),
        ({"min_count": True}, "min_count must be a positive integer"),
        ({"include_empty": "yes"}, "include_empty must be a boolean"),
    ],
)
def test_collection_member_sources_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.collection_member_sources("review", **kwargs)
