from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"Content {unit_id}",
        content_type=content_type,
    )


def test_unit_title_terms_empty_store_returns_empty(store: Store):
    assert store.unit_title_terms() == []


def test_unit_title_terms_normalizes_punctuation_case_and_repeated_terms(store: Store):
    store.insert_unit(_unit("a", "Solar, solar grid!"))
    store.insert_unit(_unit("b", "solar-grid storage"))
    store.insert_unit(_unit("c", "The grid and storage plan", source_project=SourceProject.ME))

    assert store.unit_title_terms(min_unit_count=2) == [
        {
            "term": "grid",
            "unit_count": 3,
            "source_project_counts": {"max": 2, "me": 1},
            "example_unit_ids": ["a", "b", "c"],
        },
        {
            "term": "solar",
            "unit_count": 2,
            "source_project_counts": {"max": 2},
            "example_unit_ids": ["a", "b"],
        },
        {
            "term": "storage",
            "unit_count": 2,
            "source_project_counts": {"max": 1, "me": 1},
            "example_unit_ids": ["b", "c"],
        },
    ]


def test_unit_title_terms_filters_source_project_and_content_type(store: Store):
    store.insert_unit(_unit("a", "Solar grid", source_project=SourceProject.MAX))
    store.insert_unit(_unit("b", "Solar storage", source_project=SourceProject.ME))
    store.insert_unit(_unit("c", "Grid storage", content_type=ContentType.FINDING))

    assert store.unit_title_terms(source_project=SourceProject.MAX, min_unit_count=1) == [
        {
            "term": "grid",
            "unit_count": 2,
            "source_project_counts": {"max": 2},
            "example_unit_ids": ["a", "c"],
        },
        {
            "term": "solar",
            "unit_count": 1,
            "source_project_counts": {"max": 1},
            "example_unit_ids": ["a"],
        },
        {
            "term": "storage",
            "unit_count": 1,
            "source_project_counts": {"max": 1},
            "example_unit_ids": ["c"],
        },
    ]
    assert store.unit_title_terms(content_type=ContentType.FINDING) == [
        {
            "term": "grid",
            "unit_count": 1,
            "source_project_counts": {"max": 1},
            "example_unit_ids": ["c"],
        },
        {
            "term": "storage",
            "unit_count": 1,
            "source_project_counts": {"max": 1},
            "example_unit_ids": ["c"],
        },
    ]


def test_unit_title_terms_applies_limit_and_min_term_length(store: Store):
    store.insert_unit(_unit("a", "AI map strategy"))
    store.insert_unit(_unit("b", "AI map systems"))

    assert store.unit_title_terms(min_term_length=2, limit=2) == [
        {
            "term": "ai",
            "unit_count": 2,
            "source_project_counts": {"max": 2},
            "example_unit_ids": ["a", "b"],
        },
        {
            "term": "map",
            "unit_count": 2,
            "source_project_counts": {"max": 2},
            "example_unit_ids": ["a", "b"],
        },
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_unit_count": 0}, "min_unit_count must be a positive integer"),
        ({"min_unit_count": True}, "min_unit_count must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer or None"),
        ({"limit": True}, "limit must be a positive integer or None"),
        ({"min_term_length": 0}, "min_term_length must be a positive integer"),
        ({"min_term_length": True}, "min_term_length must be a positive integer"),
    ],
)
def test_unit_title_terms_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.unit_title_terms(**kwargs)
