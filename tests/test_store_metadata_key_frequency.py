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
    title: str,
    metadata: dict,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"Content for {title}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_key_usage_frequency_empty_graph_returns_empty_dict(store: Store):
    result = store.metadata_key_usage_frequency()

    assert result == {}


def test_metadata_key_usage_frequency_single_unit_with_metadata(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"author": "Ada", "year": 2025}))

    result = store.metadata_key_usage_frequency()

    assert result == {"author": 1, "year": 1}


def test_metadata_key_usage_frequency_single_unit_without_metadata(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {}))

    result = store.metadata_key_usage_frequency()

    assert result == {}


def test_metadata_key_usage_frequency_multiple_units_with_overlapping_keys(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"author": "Ada", "year": 2025}))
    store.insert_unit(_unit("unit-2", "Beta", {"author": "Grace", "project": "Grid"}))
    store.insert_unit(_unit("unit-3", "Gamma", {"author": "Linus", "year": 2026, "tags": ["ai"]}))

    result = store.metadata_key_usage_frequency()

    assert result == {"author": 3, "year": 2, "project": 1, "tags": 1}


def test_metadata_key_usage_frequency_multiple_units_with_distinct_keys(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"author": "Ada"}))
    store.insert_unit(_unit("unit-2", "Beta", {"project": "Grid"}))
    store.insert_unit(_unit("unit-3", "Gamma", {"year": 2026}))

    result = store.metadata_key_usage_frequency()

    assert result == {"author": 1, "project": 1, "year": 1}


def test_metadata_key_usage_frequency_sorted_by_frequency_descending(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"common": "a", "rare": "b"}))
    store.insert_unit(_unit("unit-2", "Beta", {"common": "c", "medium": "d"}))
    store.insert_unit(_unit("unit-3", "Gamma", {"common": "e"}))
    store.insert_unit(_unit("unit-4", "Delta", {"common": "f", "medium": "g"}))

    result = store.metadata_key_usage_frequency()

    # Verify the order: common (4), medium (2), rare (1)
    assert list(result.keys()) == ["common", "medium", "rare"]
    assert result == {"common": 4, "medium": 2, "rare": 1}


def test_metadata_key_usage_frequency_alphabetical_order_for_ties(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"zebra": 1}))
    store.insert_unit(_unit("unit-2", "Beta", {"apple": 2}))
    store.insert_unit(_unit("unit-3", "Gamma", {"middle": 3}))

    result = store.metadata_key_usage_frequency()

    # All have frequency 1, so should be alphabetically sorted
    assert list(result.keys()) == ["apple", "middle", "zebra"]
    assert result == {"apple": 1, "middle": 1, "zebra": 1}


def test_metadata_key_usage_frequency_counts_top_level_keys_only(store: Store):
    store.insert_unit(
        _unit("unit-1", "Alpha", {"project": {"area": "grid", "score": 2}, "simple": "value"})
    )
    store.insert_unit(
        _unit("unit-2", "Beta", {"project": {"area": "storage", "owner": "ops"}})
    )

    result = store.metadata_key_usage_frequency()

    # Should only count "project" and "simple", not nested keys like "area", "score", etc.
    assert result == {"project": 2, "simple": 1}


def test_metadata_key_usage_frequency_handles_various_value_types(store: Store):
    store.insert_unit(
        _unit(
            "unit-1",
            "Alpha",
            {
                "string": "text",
                "number": 42,
                "boolean": True,
                "null": None,
                "array": [1, 2, 3],
                "object": {"nested": "data"},
            },
        )
    )

    result = store.metadata_key_usage_frequency()

    assert result == {
        "string": 1,
        "number": 1,
        "boolean": 1,
        "null": 1,
        "array": 1,
        "object": 1,
    }


def test_metadata_key_usage_frequency_mixed_metadata_presence(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha", {"author": "Ada"}))
    store.insert_unit(_unit("unit-2", "Beta", {}))
    store.insert_unit(_unit("unit-3", "Gamma", {"author": "Grace", "year": 2025}))
    store.insert_unit(_unit("unit-4", "Delta", {}))

    result = store.metadata_key_usage_frequency()

    assert result == {"author": 2, "year": 1}
