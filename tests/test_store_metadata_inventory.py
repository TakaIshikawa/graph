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
    title: str,
    metadata: dict,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type="note",
        title=title,
        content=f"Content for {title}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_key_inventory_flattens_nested_metadata_lists_and_mixed_types(
    store: Store,
):
    store.insert_unit(
        _unit(
            "unit-1",
            SourceProject.MAX,
            "a",
            "A",
            {
                "project": {"area": "grid", "score": 2},
                "authors": [{"name": "Ada"}, {"name": "Grace"}],
                "flags": [True, False],
                "empty_list": [],
            },
        )
    )
    store.insert_unit(
        _unit(
            "unit-2",
            SourceProject.FORTY_TWO,
            "b",
            "B",
            {
                "project": {"area": "grid", "score": "high"},
                "authors": [{"name": "Linus"}],
                "flags": [False],
                "empty_object": {},
            },
        )
    )
    store.insert_unit(
        _unit(
            "unit-3",
            SourceProject.MAX,
            "c",
            "C",
            {
                "project": {"area": "storage"},
                "authors": [],
                "flag": None,
            },
        )
    )

    rows = store.metadata_key_inventory()
    by_path = {row["path"]: row for row in rows}

    assert [row["path"] for row in rows] == [
        "project.area",
        "authors[0].name",
        "flags[0]",
        "project.score",
        "authors",
        "authors[1].name",
        "empty_list",
        "empty_object",
        "flag",
        "flags[1]",
    ]
    assert by_path["project.area"] == {
        "path": "project.area",
        "count": 3,
        "value_types": ["string"],
        "example_values": ["grid", "storage"],
        "source_projects": ["max", "forty_two"],
    }
    assert by_path["project.score"] == {
        "path": "project.score",
        "count": 2,
        "value_types": ["integer", "string"],
        "example_values": ["2", "high"],
        "source_projects": ["forty_two", "max"],
    }
    assert by_path["authors[0].name"] == {
        "path": "authors[0].name",
        "count": 2,
        "value_types": ["string"],
        "example_values": ["Ada", "Linus"],
        "source_projects": ["forty_two", "max"],
    }
    assert by_path["flags[0]"] == {
        "path": "flags[0]",
        "count": 2,
        "value_types": ["boolean"],
        "example_values": ["false", "true"],
        "source_projects": ["forty_two", "max"],
    }
    assert by_path["authors"] == {
        "path": "authors",
        "count": 1,
        "value_types": ["array"],
        "example_values": ["[]"],
        "source_projects": ["max"],
    }
    assert by_path["empty_list"] == {
        "path": "empty_list",
        "count": 1,
        "value_types": ["array"],
        "example_values": ["[]"],
        "source_projects": ["max"],
    }
    assert by_path["empty_object"] == {
        "path": "empty_object",
        "count": 1,
        "value_types": ["object"],
        "example_values": ["{}"],
        "source_projects": ["forty_two"],
    }
    assert by_path["flag"]["value_types"] == ["null"]
    assert by_path["flag"]["example_values"] == ["None"]


def test_metadata_key_inventory_filters_by_prefix_and_limits(store: Store):
    store.insert_unit(
        _unit(
            "unit-1",
            SourceProject.MAX,
            "a",
            "A",
            {"project": {"area": "grid", "score": 2}, "other": "ignored"},
        )
    )
    store.insert_unit(
        _unit(
            "unit-2",
            SourceProject.MAX,
            "b",
            "B",
            {"project": {"area": "storage", "owner": "ops"}},
        )
    )

    assert store.metadata_key_inventory(prefix="project", limit=2) == [
        {
            "path": "project.area",
            "count": 2,
            "value_types": ["string"],
            "example_values": ["grid", "storage"],
            "source_projects": ["max"],
        },
        {
            "path": "project.owner",
            "count": 1,
            "value_types": ["string"],
            "example_values": ["ops"],
            "source_projects": ["max"],
        },
    ]
    assert [row["path"] for row in store.metadata_key_inventory(prefix="project.s")] == [
        "project.score"
    ]


def test_metadata_key_inventory_returns_empty_list_without_metadata(store: Store):
    store.insert_unit(_unit("unit-1", SourceProject.MAX, "a", "A", {}))

    assert store.metadata_key_inventory() == []


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_metadata_key_inventory_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        store.metadata_key_inventory(limit=limit)
