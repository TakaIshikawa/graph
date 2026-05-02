from __future__ import annotations

import pytest

from graph.rag import build_result_facets


def test_build_result_facets_counts_core_facets_and_metadata_values():
    results = [
        {
            "id": "unit-a",
            "source_project": "max",
            "content_type": "insight",
            "tags": ["solar", "storage", "solar"],
            "metadata": {
                "project": {"area": "grid"},
                "priority": "high",
                "owners": ["alice", "bob", "alice"],
            },
        },
        {
            "id": "unit-b",
            "source_project": "max",
            "content_type": "finding",
            "tags": ["solar"],
            "metadata": {
                "project": {"area": "grid"},
                "priority": "low",
                "owners": ["bob"],
            },
        },
        {
            "id": "unit-c",
            "source_project": "forty_two",
            "content_type": "insight",
            "tags": ["storage"],
            "metadata": {"project": {"area": "market"}, "priority": "high"},
        },
    ]

    facets = build_result_facets(
        results,
        metadata_keys=["project.area", "priority", "owners"],
    )

    assert facets["source_project"] == [
        {"value": "max", "key": "max", "count": 2},
        {"value": "forty_two", "key": "forty_two", "count": 1},
    ]
    assert facets["content_type"] == [
        {"value": "insight", "key": "insight", "count": 2},
        {"value": "finding", "key": "finding", "count": 1},
    ]
    assert facets["tags"] == [
        {"value": "solar", "key": "solar", "count": 2},
        {"value": "storage", "key": "storage", "count": 2},
    ]
    assert facets["metadata"] == {
        "project.area": [
            {"value": "grid", "key": "grid", "count": 2},
            {"value": "market", "key": "market", "count": 1},
        ],
        "priority": [
            {"value": "high", "key": "high", "count": 2},
            {"value": "low", "key": "low", "count": 1},
        ],
        "owners": [
            {"value": "bob", "key": "bob", "count": 2},
            {"value": "alice", "key": "alice", "count": 1},
        ],
    }
    assert facets["stats"] == {
        "result_count": 3,
        "max_values": 20,
        "metadata_keys": ["project.area", "priority", "owners"],
    }


def test_build_result_facets_reads_optional_nested_unit_fields():
    results = [
        {
            "score": 0.9,
            "unit": {
                "source_project": "max",
                "content_type": "insight",
                "tags": ["solar"],
                "metadata": {"status": "approved"},
            },
        },
        {
            "source_project": "flat",
            "content_type": "finding",
            "tags": ["storage"],
            "metadata": {"status": "draft"},
            "unit": {
                "source_project": "nested",
                "content_type": "insight",
                "tags": ["solar"],
                "metadata": {"status": "approved"},
            },
        },
    ]

    facets = build_result_facets(results, metadata_keys=["status"])

    assert facets["source_project"] == [
        {"value": "flat", "key": "flat", "count": 1},
        {"value": "max", "key": "max", "count": 1},
    ]
    assert facets["content_type"] == [
        {"value": "finding", "key": "finding", "count": 1},
        {"value": "insight", "key": "insight", "count": 1},
    ]
    assert facets["tags"] == [
        {"value": "solar", "key": "solar", "count": 1},
        {"value": "storage", "key": "storage", "count": 1},
    ]
    assert facets["metadata"]["status"] == [
        {"value": "approved", "key": "approved", "count": 1},
        {"value": "draft", "key": "draft", "count": 1},
    ]


def test_build_result_facets_tolerates_missing_optional_fields():
    facets = build_result_facets(
        [
            {"id": "unit-a"},
            {"id": "unit-b", "tags": None, "metadata": None},
            {"id": "unit-c", "tags": "solar", "metadata": []},
        ],
        metadata_keys=["status"],
    )

    assert facets["source_project"] == []
    assert facets["content_type"] == []
    assert facets["tags"] == []
    assert facets["metadata"] == {"status": []}
    assert facets["stats"]["result_count"] == 3


def test_max_values_limits_each_facet_list_after_sorting():
    results = [
        {
            "source_project": "beta",
            "content_type": "finding",
            "tags": ["beta"],
            "metadata": {"status": "beta"},
        },
        {
            "source_project": "alpha",
            "content_type": "insight",
            "tags": ["alpha"],
            "metadata": {"status": "alpha"},
        },
        {
            "source_project": "alpha",
            "content_type": "metadata",
            "tags": ["gamma"],
            "metadata": {"status": "gamma"},
        },
    ]

    facets = build_result_facets(results, metadata_keys=["status"], max_values=1)

    assert facets["source_project"] == [
        {"value": "alpha", "key": "alpha", "count": 2}
    ]
    assert facets["content_type"] == [
        {"value": "finding", "key": "finding", "count": 1}
    ]
    assert facets["tags"] == [{"value": "alpha", "key": "alpha", "count": 1}]
    assert facets["metadata"]["status"] == [
        {"value": "alpha", "key": "alpha", "count": 1}
    ]
    assert facets["stats"]["max_values"] == 1


@pytest.mark.parametrize("max_values", [-1, 1.5, "2", None, True])
def test_build_result_facets_validates_max_values(max_values):
    with pytest.raises(ValueError, match="max_values must be a non-negative integer"):
        build_result_facets([], max_values=max_values)


def test_build_result_facets_is_importable_from_graph_rag():
    assert callable(build_result_facets)
