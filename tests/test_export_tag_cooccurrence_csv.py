from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_tag_cooccurrence_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_tag_cooccurrence_csv_writes_header_and_stable_ordering():
    units = [
        unit("unit-b", ["gamma", "alpha"]),
        unit("unit-a", ["beta", "alpha"]),
        unit("unit-c", ["beta", "alpha"]),
    ]

    first = export_tag_cooccurrence_csv(units)
    second = export_tag_cooccurrence_csv(reversed(units))

    assert first == second
    assert first == (
        "tag_a,tag_b,count\n"
        "alpha,beta,2\n"
        "alpha,gamma,1\n"
    )
    assert rows(first) == [
        {"tag_a": "alpha", "tag_b": "beta", "count": "2"},
        {"tag_a": "alpha", "tag_b": "gamma", "count": "1"},
    ]


def test_export_tag_cooccurrence_csv_does_not_inflate_duplicate_tags_per_unit():
    text = export_tag_cooccurrence_csv(
        [
            unit("unit-a", ["Solar", "solar", " SOLAR ", "Storage", "storage"]),
            unit("unit-b", ["solar", "storage"]),
        ]
    )

    assert rows(text) == [{"tag_a": "solar", "tag_b": "storage", "count": "2"}]


def test_export_tag_cooccurrence_csv_filters_by_min_count():
    text = export_tag_cooccurrence_csv(
        [
            unit("unit-a", ["alpha", "beta", "gamma"]),
            unit("unit-b", ["alpha", "beta"]),
            unit("unit-c", ["alpha", "gamma"]),
        ],
        min_count=2,
    )

    assert rows(text) == [
        {"tag_a": "alpha", "tag_b": "beta", "count": "2"},
        {"tag_a": "alpha", "tag_b": "gamma", "count": "2"},
    ]


def test_export_tag_cooccurrence_csv_returns_header_for_empty_input():
    assert export_tag_cooccurrence_csv([]) == "tag_a,tag_b,count\n"
