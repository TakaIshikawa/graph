from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.substack_subscribers_csv import SubstackSubscribersCsvAdapter
from graph.types.models import SyncState


def test_substack_subscribers_csv_ingests_common_headers(tmp_path):
    export = tmp_path / "subscribers.csv"
    export.write_text(
        "Email,Name,Subscription Type,Status,Pledge,Source,Subscribed At,Created At,Unsubscribed At,Stripe Customer ID\n"
        "ada@example.com,Ada Lovelace,paid,active,$5,web,2026-04-01,2026-03-31,,cus_123\n",
        encoding="utf-8",
    )

    unit = SubstackSubscribersCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "substack_subscribers_csv"
    assert unit.source_entity_type == "subscriber"
    assert unit.source_id == "substack_subscribers_csv:ada@example.com"
    assert unit.metadata["email"] == "ada@example.com"
    assert unit.metadata["subscription_type"] == "paid"
    assert unit.metadata["stripe_customer_id"] == "cus_123"
    assert "Email: ada@example.com" in unit.content
    assert "paid" in unit.tags


def test_substack_subscribers_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "subscribers.csv"
    export.write_text(
        "Email Address,Subscriber Name,Plan,Subscription Status,Signup Source,Joined At\n"
        "new@example.com,New Reader,free,active,import,2026-04-03\n"
        "old@example.com,Old Reader,paid,canceled,web,2026-04-01\n"
        ",,,,,\n"
        ",Named Only,,,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="substack_subscribers_csv", source_entity_type="subscriber", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = SubstackSubscribersCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["email"] for unit in units] == ["new@example.com"]
    all_units = SubstackSubscribersCsvAdapter(path=str(export)).ingest().units
    assert [unit.title for unit in all_units] == ["Old Reader", "New Reader", "Named Only"]
    assert SubstackSubscribersCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
