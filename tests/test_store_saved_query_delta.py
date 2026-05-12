from __future__ import annotations

import pytest

from graph.store.db import Store


def test_compare_saved_query_runs_reports_added_removed_and_retained(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    try:
        store.save_query(name="solar", query="solar", mode="fulltext", limit=10)
        left = store.record_saved_query_run(
            "solar",
            run_at="2026-05-01T00:00:00+00:00",
            effective_limit=10,
            mode="fulltext",
            filters={},
            result_count=3,
            top_result_ids=["c", "a", "b"],
        )
        right = store.record_saved_query_run(
            "solar",
            run_at="2026-05-02T00:00:00+00:00",
            effective_limit=10,
            mode="fulltext",
            filters={},
            result_count=3,
            top_result_ids=["b", "d", "a"],
        )

        delta = store.compare_saved_query_runs(left["id"], right["id"])

        assert delta["saved_query_name"] == "solar"
        assert delta["left_run"]["id"] == left["id"]
        assert delta["right_run"]["id"] == right["id"]
        assert delta["added"] == ["d"]
        assert delta["removed"] == ["c"]
        assert delta["retained"] == ["a", "b"]
        assert delta["added_count"] == 1
        assert delta["removed_count"] == 1
        assert delta["retained_count"] == 2
        assert delta["unchanged_count"] == 2
    finally:
        store.close()


def test_compare_saved_query_runs_rejects_unknown_run_ids(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    try:
        with pytest.raises(ValueError, match="Saved query run not found"):
            store.compare_saved_query_runs(1, 2)
    finally:
        store.close()


def test_compare_saved_query_runs_rejects_different_saved_query_names(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    try:
        store.save_query(name="solar", query="solar", mode="fulltext", limit=10)
        store.save_query(name="battery", query="battery", mode="fulltext", limit=10)
        left = store.record_saved_query_run(
            "solar",
            effective_limit=10,
            mode="fulltext",
            filters={},
            result_count=1,
            top_result_ids=["a"],
        )
        right = store.record_saved_query_run(
            "battery",
            effective_limit=10,
            mode="fulltext",
            filters={},
            result_count=1,
            top_result_ids=["a"],
        )

        with pytest.raises(ValueError, match="same saved query"):
            store.compare_saved_query_runs(left["id"], right["id"])
    finally:
        store.close()
