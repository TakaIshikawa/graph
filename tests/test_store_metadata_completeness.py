"""Tests for store metadata completeness summaries."""

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
    unit_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        metadata=metadata or {},
    )


def test_metadata_completeness_summary_counts_flat_nested_and_missing_keys(store: Store):
    store.insert_unit(
        unit(
            "unit-a",
            metadata={
                "author": "A. Smith",
                "review": {"state": "approved"},
                "empty_string": " ",
                "false_is_present": False,
            },
        )
    )
    store.insert_unit(
        unit(
            "unit-b",
            metadata={
                "author": "",
                "review": {"state": "draft"},
                "empty_list": [],
                "false_is_present": False,
            },
        )
    )
    store.insert_unit(
        unit(
            "unit-c",
            metadata={
                "author": "C. Lee",
                "review": {},
                "empty_string": "value",
            },
        )
    )

    summary = store.metadata_completeness_summary(
        [
            "review.state",
            "author",
            "empty_string",
            "empty_list",
            "false_is_present",
        ]
    )

    assert summary == {
        "total_units": 3,
        "required_keys": [
            "author",
            "empty_list",
            "empty_string",
            "false_is_present",
            "review.state",
        ],
        "source_project": None,
        "content_type": None,
        "keys": [
            {
                "key": "author",
                "present_count": 2,
                "missing_count": 1,
                "missing_unit_ids": ["unit-b"],
            },
            {
                "key": "empty_list",
                "present_count": 0,
                "missing_count": 3,
                "missing_unit_ids": ["unit-a", "unit-b", "unit-c"],
            },
            {
                "key": "empty_string",
                "present_count": 1,
                "missing_count": 2,
                "missing_unit_ids": ["unit-a", "unit-b"],
            },
            {
                "key": "false_is_present",
                "present_count": 2,
                "missing_count": 1,
                "missing_unit_ids": ["unit-c"],
            },
            {
                "key": "review.state",
                "present_count": 2,
                "missing_count": 1,
                "missing_unit_ids": ["unit-c"],
            },
        ],
        "present_counts": {
            "author": 2,
            "empty_list": 0,
            "empty_string": 1,
            "false_is_present": 2,
            "review.state": 2,
        },
        "missing_counts": {
            "author": 1,
            "empty_list": 3,
            "empty_string": 2,
            "false_is_present": 1,
            "review.state": 1,
        },
        "missing_unit_ids": {
            "author": ["unit-b"],
            "empty_list": ["unit-a", "unit-b", "unit-c"],
            "empty_string": ["unit-a", "unit-b"],
            "false_is_present": ["unit-c"],
            "review.state": ["unit-c"],
        },
    }


def test_metadata_completeness_summary_filters_by_source_project_and_content_type(
    store: Store,
):
    store.insert_unit(
        unit(
            "max-insight-a",
            source_project=SourceProject.MAX,
            content_type=ContentType.INSIGHT,
            metadata={"doi": "10.1000/a"},
        )
    )
    store.insert_unit(
        unit(
            "max-insight-b",
            source_project=SourceProject.MAX,
            content_type=ContentType.INSIGHT,
            metadata={},
        )
    )
    store.insert_unit(
        unit(
            "max-finding",
            source_project=SourceProject.MAX,
            content_type=ContentType.FINDING,
            metadata={},
        )
    )
    store.insert_unit(
        unit(
            "forty-two-insight",
            source_project=SourceProject.FORTY_TWO,
            content_type=ContentType.INSIGHT,
            metadata={},
        )
    )

    summary = store.metadata_completeness_summary(
        ["doi"],
        source_project="max",
        content_type="insight",
    )

    assert summary["total_units"] == 2
    assert summary["source_project"] == "max"
    assert summary["content_type"] == "insight"
    assert summary["present_counts"] == {"doi": 1}
    assert summary["missing_counts"] == {"doi": 1}
    assert summary["missing_unit_ids"] == {"doi": ["max-insight-b"]}


def test_metadata_completeness_summary_empty_requirements_counts_filtered_units(
    store: Store,
):
    store.insert_unit(unit("unit-a", source_project=SourceProject.MAX))
    store.insert_unit(unit("unit-b", source_project=SourceProject.FORTY_TWO))

    summary = store.metadata_completeness_summary([], source_project="max")

    assert summary == {
        "total_units": 1,
        "required_keys": [],
        "source_project": "max",
        "content_type": None,
        "keys": [],
        "present_counts": {},
        "missing_counts": {},
        "missing_unit_ids": {},
    }


def test_metadata_completeness_summary_empty_store_reports_zero_counts(store: Store):
    summary = store.metadata_completeness_summary(["author", "review.state"])

    assert summary["total_units"] == 0
    assert summary["required_keys"] == ["author", "review.state"]
    assert summary["present_counts"] == {"author": 0, "review.state": 0}
    assert summary["missing_counts"] == {"author": 0, "review.state": 0}
    assert summary["missing_unit_ids"] == {"author": [], "review.state": []}


def test_metadata_completeness_summary_rejects_invalid_dotted_keys(store: Store):
    with pytest.raises(ValueError, match="non-empty dotted path"):
        store.metadata_completeness_summary(["review."])
