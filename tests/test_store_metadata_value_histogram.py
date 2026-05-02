from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    for candidate in (
        Path(path),
        Path(path).with_name(Path(path).name + "-wal"),
        Path(path).with_name(Path(path).name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def unit(
    source_id: str,
    metadata: dict,
    *,
    source_project: SourceProject = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="insight",
        title=f"Unit {source_id}",
        content=f"Content {source_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_value_histogram_counts_nested_scalar_values_and_missing(store: Store):
    store.insert_unit(unit("a", {"review": {"state": "approved"}}))
    store.insert_unit(unit("b", {"review": {"state": "draft"}}))
    store.insert_unit(unit("c", {"review": {"state": "approved"}}))
    store.insert_unit(unit("d", {"review": {"priority": "high"}}))
    store.insert_unit(unit("e", {}))

    assert store.metadata_value_histogram("review.state") == {
        "path": "review.state",
        "source_project": None,
        "unit_count": 5,
        "missing_count": 2,
        "value_count": 3,
        "values": [
            {"value": "approved", "value_type": "string", "count": 2},
            {"value": "draft", "value_type": "string", "count": 1},
        ],
    }


def test_metadata_value_histogram_counts_list_values_per_item(store: Store):
    store.insert_unit(unit("a", {"project": {"areas": ["solar", "grid"]}}))
    store.insert_unit(unit("b", {"project": {"areas": ["solar", "storage", "solar"]}}))
    store.insert_unit(unit("c", {"project": {"areas": []}}))
    store.insert_unit(unit("d", {"project": {"area": "solar"}}))

    assert store.metadata_value_histogram("project.areas") == {
        "path": "project.areas",
        "source_project": None,
        "unit_count": 4,
        "missing_count": 1,
        "value_count": 5,
        "values": [
            {"value": "solar", "value_type": "string", "count": 3},
            {"value": "grid", "value_type": "string", "count": 1},
            {"value": "storage", "value_type": "string", "count": 1},
        ],
    }


def test_metadata_value_histogram_supports_source_project_filter_and_limit(store: Store):
    store.insert_unit(
        unit("max-a", {"review": {"state": "approved"}}, source_project=SourceProject.MAX)
    )
    store.insert_unit(
        unit("max-b", {"review": {"state": "approved"}}, source_project=SourceProject.MAX)
    )
    store.insert_unit(
        unit("max-c", {"review": {"state": "draft"}}, source_project=SourceProject.MAX)
    )
    store.insert_unit(
        unit(
            "forty-two-a",
            {"review": {"state": "archived"}},
            source_project=SourceProject.FORTY_TWO,
        )
    )

    assert store.metadata_value_histogram(
        "review.state",
        source_project="max",
        limit=1,
    ) == {
        "path": "review.state",
        "source_project": "max",
        "unit_count": 3,
        "missing_count": 0,
        "value_count": 3,
        "values": [{"value": "approved", "value_type": "string", "count": 2}],
    }


def test_metadata_value_histogram_keeps_json_scalar_types_distinct(store: Store):
    store.insert_unit(unit("a", {"flags": {"enabled": True, "rank": 1, "note": None}}))
    store.insert_unit(unit("b", {"flags": {"enabled": 1, "rank": 1, "note": None}}))

    assert store.metadata_value_histogram("flags.enabled")["values"] == [
        {"value": True, "value_type": "boolean", "count": 1},
        {"value": 1, "value_type": "integer", "count": 1},
    ]
    assert store.metadata_value_histogram("flags.note")["values"] == [
        {"value": None, "value_type": "null", "count": 2}
    ]


def test_metadata_value_histogram_validates_path_and_limit(store: Store):
    with pytest.raises(ValueError, match="positive integer"):
        store.metadata_value_histogram("review.state", limit=0)

    with pytest.raises(ValueError, match="non-empty dotted path"):
        store.metadata_value_histogram("review.")
