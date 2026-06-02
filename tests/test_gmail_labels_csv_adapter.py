from datetime import datetime, timezone

from graph.adapters.gmail_labels_csv import GmailLabelsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_gmail_labels_csv_ingests_labels_and_participants(tmp_path):
    path = tmp_path / "gmail.csv"
    path.write_text("Message ID,Thread ID,Subject,From,To,Date,Labels,Snippet,URL\nm1,t1,Hello,a@example.com,b@example.com,2026-05-02T00:00:00Z,\"Inbox;Work\",Preview,https://mail.test/m1\n", encoding="utf-8")

    unit = GmailLabelsCsvAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "gmail_labels_csv:m1"
    assert unit.metadata["thread_id"] == "t1"
    assert unit.metadata["from"] == "a@example.com"
    assert unit.metadata["labels"] == ["Inbox", "Work"]
    assert {"gmail", "email_label", "Inbox", "Work"}.issubset(set(unit.tags))


def test_gmail_labels_csv_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "gmail.csv"
    path.write_text("message id,subject,date\nold,Old,2026-04-01T00:00:00Z\nnew,New,2026-05-02T00:00:00Z\n", encoding="utf-8")
    since = SyncState(source_project="gmail_labels_csv", source_entity_type="email_label", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = GmailLabelsCsvAdapter(str(path)).ingest(since=since, entity_types=["email_label"])

    assert [unit.source_id for unit in result.units] == ["gmail_labels_csv:new"]
    assert GmailLabelsCsvAdapter(str(path)).ingest(entity_types=["message"]).units == []
    assert get_adapter("gmail_labels_csv", path=str(path)).name == "gmail_labels_csv"
