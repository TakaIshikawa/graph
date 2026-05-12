from __future__ import annotations

from datetime import datetime, timezone

from graph.store.db import Store


def test_list_due_saved_queries_returns_scheduled_queries_due_now(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    now = datetime(2026, 5, 12, 12, tzinfo=timezone.utc)
    try:
        store.save_query(name="daily due", query="alpha", schedule="daily")
        store.save_query(name="daily fresh", query="beta", schedule="daily")
        store.save_query(name="unscheduled", query="gamma")
        store.conn.execute(
            "UPDATE saved_queries SET last_run_at = ? WHERE name = ?",
            ("2026-05-10T12:00:00+00:00", "daily due"),
        )
        store.conn.execute(
            "UPDATE saved_queries SET last_run_at = ? WHERE name = ?",
            ("2026-05-12T00:00:00+00:00", "daily fresh"),
        )
        store.conn.commit()

        assert [row["name"] for row in store.list_due_saved_queries(now)] == ["daily due"]
    finally:
        store.close()


def test_list_due_saved_queries_treats_never_run_scheduled_queries_as_due(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.save_query(name="weekly never run", query="alpha", schedule="weekly")

        due = store.list_due_saved_queries(datetime(2026, 5, 12, tzinfo=timezone.utc))

        assert [row["name"] for row in due] == ["weekly never run"]
        assert due[0] == store.list_saved_queries()[0]
    finally:
        store.close()


def test_list_due_saved_queries_supports_weekly_monthly_and_naive_now(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.save_query(name="monthly due", query="alpha", schedule="monthly")
        store.save_query(name="weekly due", query="beta", schedule="weekly")
        store.save_query(name="monthly fresh", query="gamma", schedule="monthly")
        updates = [
            ("2026-04-01T00:00:00+00:00", "monthly due"),
            ("2026-05-01T00:00:00+00:00", "weekly due"),
            ("2026-04-20T00:00:00+00:00", "monthly fresh"),
        ]
        store.conn.executemany(
            "UPDATE saved_queries SET last_run_at = ? WHERE name = ?",
            updates,
        )
        store.conn.commit()

        due = store.list_due_saved_queries(datetime(2026, 5, 12, 0, 0, 0))

        assert [row["name"] for row in due] == ["monthly due", "weekly due"]
    finally:
        store.close()
