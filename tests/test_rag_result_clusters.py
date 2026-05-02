from __future__ import annotations

import pytest

from graph.rag import cluster_results_by_overlap
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def test_cluster_results_by_overlap_groups_results_with_shared_tags():
    results = [
        {
            "id": "unit-a",
            "title": "Solar storage roadmap",
            "source_project": "max",
            "tags": ["Solar", "storage"],
        },
        {
            "id": "unit-b",
            "title": "Storage market notes",
            "source_project": "presence",
            "tags": ["storage", "market"],
        },
        {
            "id": "unit-c",
            "title": "Async service design",
            "source_project": "max",
            "tags": ["python"],
        },
    ]

    assert cluster_results_by_overlap(results) == [
        {
            "id": "cluster-1-storage-solar",
            "label": "storage + Solar",
            "size": 2,
            "tags": ["storage", "Solar", "market"],
            "sources": ["max", "presence"],
            "result_ids": ["unit-a", "unit-b"],
            "representative_title": "Solar storage roadmap",
        },
        {
            "id": "cluster-2-python",
            "label": "python",
            "size": 1,
            "tags": ["python"],
            "sources": ["max"],
            "result_ids": ["unit-c"],
            "representative_title": "Async service design",
        },
    ]


def test_cluster_results_by_overlap_uses_min_shared_tags():
    results = [
        {"id": "unit-a", "source_project": "max", "tags": ["solar", "storage"]},
        {"id": "unit-b", "source_project": "presence", "tags": ["solar", "market"]},
    ]

    assert cluster_results_by_overlap(results, min_shared_tags=2) == [
        {
            "id": "cluster-1-solar-storage",
            "label": "solar + storage",
            "size": 1,
            "tags": ["solar", "storage"],
            "sources": ["max"],
            "result_ids": ["unit-a"],
            "representative_title": None,
        },
        {
            "id": "cluster-2-market-solar",
            "label": "market + solar",
            "size": 1,
            "tags": ["market", "solar"],
            "sources": ["presence"],
            "result_ids": ["unit-b"],
            "representative_title": None,
        },
    ]


def test_cluster_results_by_overlap_groups_tagless_results_by_source():
    results = [
        {"id": "unit-b", "title": "Second", "source_project": "max", "tags": []},
        {"id": "unit-a", "title": "First", "source_project": "max"},
        {"id": "unit-c", "title": "Other", "source_project": "presence"},
    ]

    assert cluster_results_by_overlap(results) == [
        {
            "id": "cluster-1-max",
            "label": "max",
            "size": 2,
            "tags": [],
            "sources": ["max"],
            "result_ids": ["unit-a", "unit-b"],
            "representative_title": "First",
        },
        {
            "id": "cluster-2-presence",
            "label": "presence",
            "size": 1,
            "tags": [],
            "sources": ["presence"],
            "result_ids": ["unit-c"],
            "representative_title": "Other",
        },
    ]


def test_cluster_results_by_overlap_reads_nested_unit_objects_and_flat_fields_win():
    results = [
        {"score": 0.9, "unit": unit("unit-a", "Nested A", tags=["solar"])},
        {"score": 0.8, "unit": unit("unit-b", "Nested B", tags=["solar"])},
        {
            "id": "flat-id",
            "title": "Flat title",
            "source_project": "flat",
            "tags": ["flat-tag"],
            "unit": unit("unit-c", "Nested C", tags=["solar"]),
        },
    ]

    clusters = cluster_results_by_overlap(results)

    assert clusters == [
        {
            "id": "cluster-1-solar",
            "label": "solar",
            "size": 2,
            "tags": ["solar"],
            "sources": ["max"],
            "result_ids": ["unit-a", "unit-b"],
            "representative_title": "Nested A",
        },
        {
            "id": "cluster-2-flat-tag",
            "label": "flat-tag",
            "size": 1,
            "tags": ["flat-tag"],
            "sources": ["flat"],
            "result_ids": ["flat-id"],
            "representative_title": "Flat title",
        },
    ]


def test_cluster_results_by_overlap_limits_clusters_without_dropping_results():
    results = [
        {"id": "unit-a", "source_project": "alpha", "tags": ["a"]},
        {"id": "unit-b", "source_project": "beta", "tags": ["b"]},
        {"id": "unit-c", "source_project": "gamma", "tags": ["c"]},
    ]

    clusters = cluster_results_by_overlap(results, max_clusters=2)

    assert len(clusters) == 2
    assert clusters[0]["result_ids"] == ["unit-a"]
    assert clusters[1]["size"] == 2
    assert clusters[1]["result_ids"] == ["unit-b", "unit-c"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_clusters": 0}, "max_clusters"),
        ({"max_clusters": -1}, "max_clusters"),
        ({"max_clusters": "2"}, "max_clusters"),
        ({"max_clusters": True}, "max_clusters"),
        ({"min_shared_tags": 0}, "min_shared_tags"),
        ({"min_shared_tags": 1.5}, "min_shared_tags"),
        ({"min_shared_tags": None}, "min_shared_tags"),
        ({"min_shared_tags": False}, "min_shared_tags"),
    ],
)
def test_cluster_results_by_overlap_validates_arguments(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        cluster_results_by_overlap([], **kwargs)


def test_cluster_results_by_overlap_is_importable_from_graph_rag():
    assert callable(cluster_results_by_overlap)
