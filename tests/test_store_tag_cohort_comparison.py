from __future__ import annotations

import os
import tempfile

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
    os.unlink(path)


def unit(
    unit_id: str,
    tags: list[str],
    *,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags,
        metadata=metadata or {},
    )


def test_compare_tag_cohorts_reports_overlap_and_distributions(store: Store):
    store.insert_unit(
        unit("a", ["solar", "storage"], metadata={"review": {"state": "approved"}})
    )
    store.insert_unit(
        unit(
            "b",
            ["solar"],
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.FINDING,
            metadata={"owner": "left"},
        )
    )
    store.insert_unit(unit("c", ["storage"], metadata={"source": "right"}))

    result = store.compare_tag_cohorts("solar", "storage")

    assert result["left_count"] == 2
    assert result["right_count"] == 2
    assert result["overlap_count"] == 1
    assert [unit["id"] for unit in result["overlap_units"]] == ["a"]
    assert [unit["id"] for unit in result["left_only"]] == ["b"]
    assert [unit["id"] for unit in result["right_only"]] == ["c"]
    assert result["source_project_distributions"] == {
        "left": {"max": 1, "presence": 1},
        "right": {"max": 2},
    }
    assert result["content_type_distributions"]["left"] == {"finding": 1, "insight": 1}
    assert result["metadata_key_differences"] == {
        "left_only": ["owner"],
        "right_only": ["source"],
        "shared": ["review.state"],
    }


def test_compare_tag_cohorts_handles_disjoint_and_empty_cohorts(store: Store):
    store.insert_unit(unit("a", ["alpha"]))
    store.insert_unit(unit("b", ["beta"]))

    disjoint = store.compare_tag_cohorts("alpha", "beta")
    assert disjoint["overlap_units"] == []
    assert [unit["id"] for unit in disjoint["left_only"]] == ["a"]
    assert [unit["id"] for unit in disjoint["right_only"]] == ["b"]

    empty = store.compare_tag_cohorts("missing-left", "missing-right")
    assert empty["left_count"] == 0
    assert empty["right_count"] == 0
    assert empty["source_project_distributions"] == {"left": {}, "right": {}}


def test_compare_tag_cohorts_applies_limit_deterministically(store: Store):
    for unit_id in ["c", "a", "b"]:
        store.insert_unit(unit(unit_id, ["left"]))
    for unit_id in ["e", "d"]:
        store.insert_unit(unit(unit_id, ["right"]))

    result = store.compare_tag_cohorts("left", "right", limit=2)

    assert [unit["id"] for unit in result["left_only"]] == ["a", "b"]
    assert [unit["id"] for unit in result["right_only"]] == ["d", "e"]


def test_compare_tag_cohorts_uses_exact_tag_matching(store: Store):
    store.insert_unit(unit("a", ["solar"]))
    store.insert_unit(unit("b", ["solar-power"]))

    result = store.compare_tag_cohorts("solar", "solar-power")

    assert result["left_count"] == 1
    assert result["right_count"] == 1
    assert result["overlap_count"] == 0
