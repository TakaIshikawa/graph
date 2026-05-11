from __future__ import annotations

import math
import os
import tempfile

import pytest

from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        tags=tags,
    )


def test_compute_tag_cooccurrence_matrix_counts_pairs(store: Store):
    store.insert_unit(unit("a", ["solar", "storage", "grid"]))
    store.insert_unit(unit("b", ["solar", "storage"]))
    store.insert_unit(unit("c", ["solar"]))

    assert store.compute_tag_cooccurrence_matrix() == {
        ("solar", "storage"): 2,
        ("grid", "solar"): 1,
        ("grid", "storage"): 1,
    }


def test_compute_tag_cooccurrence_matrix_filters_by_min_count(store: Store):
    store.insert_unit(unit("a", ["solar", "storage", "grid"]))
    store.insert_unit(unit("b", ["solar", "storage"]))

    assert store.compute_tag_cooccurrence_matrix(min_count=2) == {
        ("solar", "storage"): 2
    }


def test_compute_tag_cooccurrence_matrix_supports_normalization_modes(store: Store):
    store.insert_unit(unit("a", ["solar", "storage"]))
    store.insert_unit(unit("b", ["solar", "storage"]))
    store.insert_unit(unit("c", ["solar", "grid"]))

    jaccard = store.compute_tag_cooccurrence_matrix(normalization="jaccard")
    cosine = store.compute_tag_cooccurrence_matrix(normalization="cosine")

    assert jaccard[("solar", "storage")] == pytest.approx(2 / 3)
    assert cosine[("solar", "storage")] == pytest.approx(2 / math.sqrt(3 * 2))


def test_get_tag_cooccurrence_matrix_alias_and_large_dataset(store: Store):
    for index in range(120):
        tags = ["common", f"group-{index % 4}"]
        if index % 3 == 0:
            tags.append("triad")
        store.insert_unit(unit(f"unit-{index:03d}", tags))

    matrix = store.get_tag_cooccurrence_matrix(min_count=30)

    assert matrix[("common", "group-0")] == 30
    assert matrix[("common", "group-1")] == 30
    assert matrix[("common", "triad")] == 40


def test_compute_tag_cooccurrence_matrix_validates_arguments(store: Store):
    with pytest.raises(ValueError, match="min_count"):
        store.compute_tag_cooccurrence_matrix(min_count=0)
    with pytest.raises(ValueError, match="normalization"):
        store.compute_tag_cooccurrence_matrix(normalization="bad")
