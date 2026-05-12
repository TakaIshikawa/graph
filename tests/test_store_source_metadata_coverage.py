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
    source_entity_type: str,
    metadata: dict,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=unit_id,
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_source_metadata_coverage_reports_per_source_entity_path_rows(store: Store):
    store.insert_unit(
        _unit(
            "a1",
            SourceProject.MAX,
            "note",
            {
                "project": {"area": "grid", "score": 2},
                "authors": [{"name": "Ada"}],
                "empty": "",
            },
        )
    )
    store.insert_unit(
        _unit(
            "a2",
            SourceProject.MAX,
            "note",
            {
                "project": {"area": "storage"},
                "authors": [],
                "empty": None,
            },
        )
    )
    store.insert_unit(
        _unit(
            "b1",
            SourceProject.FORTY_TWO,
            "task",
            {"project": {"area": "grid"}, "done": True},
        )
    )

    rows = store.source_metadata_coverage()
    by_key = {
        (row["source_project"], row["source_entity_type"], row["metadata_path"]): row
        for row in rows
    }

    assert rows == sorted(
        rows,
        key=lambda row: (
            row["source_project"],
            row["source_entity_type"],
            row["metadata_path"],
        ),
    )
    assert by_key[("max", "note", "project.area")] == {
        "source_project": "max",
        "source_entity_type": "note",
        "metadata_path": "project.area",
        "present_count": 2,
        "total_unit_count": 2,
        "coverage_ratio": 1.0,
        "value_type_distribution": {"string": 2},
        "sample_values": ["grid", "storage"],
    }
    assert by_key[("max", "note", "project.score")]["present_count"] == 1
    assert by_key[("max", "note", "project.score")]["coverage_ratio"] == 0.5
    assert by_key[("max", "note", "authors[0].name")]["sample_values"] == ["Ada"]
    assert by_key[("max", "note", "authors")]["value_type_distribution"] == {"array": 1}
    assert by_key[("max", "note", "authors")]["present_count"] == 0
    assert by_key[("max", "note", "empty")]["value_type_distribution"] == {
        "null": 1,
        "string": 1,
    }
    assert by_key[("max", "note", "empty")]["present_count"] == 0
    assert by_key[("forty_two", "task", "done")]["present_count"] == 1


def test_source_metadata_coverage_filters_and_explicit_paths_keep_shape(store: Store):
    store.insert_unit(
        _unit(
            "a1",
            "max",
            "note",
            {"project": {"area": "grid"}, "quality": {"score": 4}},
        )
    )
    store.insert_unit(_unit("a2", "max", "note", {"quality": {"score": 5}}))
    store.insert_unit(_unit("b1", "max", "task", {"project": {"area": "ops"}}))
    store.insert_unit(_unit("c1", "readwise", "note", {"project": {"area": "book"}}))

    rows = store.source_metadata_coverage(
        source_project="max",
        source_entity_type="note",
        metadata_paths=["project.area", "missing.path"],
        sample_limit=1,
    )

    assert rows == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_path": "project.area",
            "present_count": 1,
            "total_unit_count": 2,
            "coverage_ratio": 0.5,
            "value_type_distribution": {"string": 1},
            "sample_values": ["grid"],
        },
        {
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_path": "missing.path",
            "present_count": 0,
            "total_unit_count": 2,
            "coverage_ratio": 0.0,
            "value_type_distribution": {},
            "sample_values": [],
        },
    ]


def test_source_metadata_coverage_returns_empty_list_without_units(store: Store):
    assert store.source_metadata_coverage(metadata_paths=["project.area"]) == []


@pytest.mark.parametrize("sample_limit", [-1, 1.5, True])
def test_source_metadata_coverage_validates_sample_limit(store: Store, sample_limit):
    with pytest.raises(ValueError, match="sample_limit must be a non-negative integer"):
        store.source_metadata_coverage(sample_limit=sample_limit)


@pytest.mark.parametrize("metadata_paths", ["project.area", [""]])
def test_source_metadata_coverage_validates_metadata_paths(store: Store, metadata_paths):
    with pytest.raises(ValueError, match="metadata_paths must be a sequence"):
        store.source_metadata_coverage(metadata_paths=metadata_paths)
