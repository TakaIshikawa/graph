from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def test_collection_overlap_returns_pairwise_metrics_in_deterministic_order(
    store: Store,
):
    result = GraphService(store).collection_overlap(
        {
            "zeta": ["unit-1", "unit-2", "unit-3", "unit-4"],
            "alpha": ["unit-2", "unit-3", "unit-4", "unit-5"],
            "beta": ["unit-2", "unit-3", "unit-6"],
        }
    )

    assert result == [
        {
            "left_collection": "alpha",
            "right_collection": "zeta",
            "overlap_size": 3,
            "jaccard": 0.6,
            "left_only_count": 1,
            "right_only_count": 1,
            "shared_unit_ids": ["unit-2", "unit-3", "unit-4"],
        },
        {
            "left_collection": "alpha",
            "right_collection": "beta",
            "overlap_size": 2,
            "jaccard": 0.4,
            "left_only_count": 2,
            "right_only_count": 1,
            "shared_unit_ids": ["unit-2", "unit-3"],
        },
        {
            "left_collection": "beta",
            "right_collection": "zeta",
            "overlap_size": 2,
            "jaccard": 0.4,
            "left_only_count": 1,
            "right_only_count": 2,
            "shared_unit_ids": ["unit-2", "unit-3"],
        },
    ]


def test_collection_overlap_counts_duplicate_unit_ids_once(store: Store):
    result = GraphService(store).collection_overlap(
        {
            "left": ["unit-1", "unit-1", "unit-2"],
            "right": ["unit-1", "unit-3", "unit-3"],
        }
    )

    assert result == [
        {
            "left_collection": "left",
            "right_collection": "right",
            "overlap_size": 1,
            "jaccard": 0.333333,
            "left_only_count": 1,
            "right_only_count": 1,
            "shared_unit_ids": ["unit-1"],
        }
    ]


def test_collection_overlap_handles_empty_collections(store: Store):
    result = GraphService(store).collection_overlap(
        {
            "empty-a": [],
            "empty-b": [],
            "non-empty": ["unit-1"],
        },
        min_overlap=0,
    )

    assert result == [
        {
            "left_collection": "empty-a",
            "right_collection": "empty-b",
            "overlap_size": 0,
            "jaccard": 0.0,
            "left_only_count": 0,
            "right_only_count": 0,
            "shared_unit_ids": [],
        },
        {
            "left_collection": "empty-a",
            "right_collection": "non-empty",
            "overlap_size": 0,
            "jaccard": 0.0,
            "left_only_count": 0,
            "right_only_count": 1,
            "shared_unit_ids": [],
        },
        {
            "left_collection": "empty-b",
            "right_collection": "non-empty",
            "overlap_size": 0,
            "jaccard": 0.0,
            "left_only_count": 0,
            "right_only_count": 1,
            "shared_unit_ids": [],
        },
    ]


def test_collection_overlap_filters_by_min_overlap(store: Store):
    result = GraphService(store).collection_overlap(
        {
            "alpha": ["unit-1", "unit-2", "unit-3"],
            "beta": ["unit-2", "unit-3", "unit-4"],
            "gamma": ["unit-3", "unit-5"],
        },
        min_overlap=2,
    )

    assert [(item["left_collection"], item["right_collection"]) for item in result] == [
        ("alpha", "beta")
    ]
    assert result[0]["overlap_size"] == 2


@pytest.mark.parametrize("min_overlap", [-1, 1.5, True, "2"])
def test_collection_overlap_rejects_invalid_min_overlap(store: Store, min_overlap):
    with pytest.raises(ValueError, match="min_overlap must be a non-negative integer"):
        GraphService(store).collection_overlap({"alpha": [], "beta": []}, min_overlap=min_overlap)
