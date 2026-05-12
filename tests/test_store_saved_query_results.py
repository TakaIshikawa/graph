from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def test_store_saves_lists_and_loads_query_results(store: Store):
    created = datetime(2026, 5, 1, tzinfo=timezone.utc)

    saved = store.save_query_results(
        name="recent-design-notes",
        query_text="design notes",
        unit_ids=["unit-b", "unit-a"],
        source_filters={"source_project": "max"},
        parameters={"limit": 25},
        metadata={"owner": "analysis"},
        created_at=created,
    )

    assert saved["id"] > 0
    assert saved["name"] == "recent-design-notes"
    assert saved["unit_ids"] == ["unit-b", "unit-a"]
    assert saved["unit_count"] == 2
    assert saved["created_at"] == "2026-05-01T00:00:00+00:00"

    summaries = store.list_saved_query_results()
    assert summaries == [
        {
            "id": saved["id"],
            "name": "recent-design-notes",
            "query_text": "design notes",
            "unit_count": 2,
            "source_filters": {"source_project": "max"},
            "parameters": {"limit": 25},
            "metadata": {"owner": "analysis"},
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": saved["updated_at"],
        }
    ]
    assert "unit_ids" not in summaries[0]
    assert store.get_saved_query_result(name="recent-design-notes") == saved
    assert store.get_saved_query_result(result_id=saved["id"]) == saved


def test_store_saved_query_results_duplicate_names_require_replace(store: Store):
    original = store.save_query_results(name="same", query_text="old", unit_ids=["a"])

    with pytest.raises(ValueError, match="already exists"):
        store.save_query_results(name="same", query_text="new", unit_ids=["b"])

    replaced = store.save_query_results(
        name="same",
        query_text="new",
        unit_ids=["b", "c"],
        replace=True,
    )

    assert replaced["id"] == original["id"]
    assert replaced["query_text"] == "new"
    assert replaced["unit_ids"] == ["b", "c"]
    assert replaced["created_at"] == original["created_at"]


def test_store_saved_query_results_delete_by_name_or_id(store: Store):
    first = store.save_query_results(name="first", unit_ids=["a"])
    second = store.save_query_results(name="second", unit_ids=["b"])

    assert store.delete_saved_query_result(name="first") is True
    assert store.get_saved_query_result(result_id=first["id"]) is None
    assert store.delete_saved_query_result(result_id=second["id"]) is True
    assert store.list_saved_query_results() == []
    assert store.delete_saved_query_result(name="missing") is False


def test_store_saved_query_results_validates_inputs(store: Store):
    with pytest.raises(ValueError, match="name must be non-empty"):
        store.save_query_results(name=" ", unit_ids=[])
    with pytest.raises(ValueError, match="unit_ids must be"):
        store.save_query_results(name="bad", unit_ids="unit-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON-serializable"):
        store.save_query_results(name="bad", unit_ids=[], metadata={"bad": object()})
    with pytest.raises(ValueError, match="exactly one"):
        store.get_saved_query_result()
    with pytest.raises(ValueError, match="exactly one"):
        store.delete_saved_query_result(name="a", result_id=1)
