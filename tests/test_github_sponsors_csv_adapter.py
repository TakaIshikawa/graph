from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.github_sponsors_csv import GitHubSponsorsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_github_sponsors_csv_current_ended_sparse_filters_and_registry(tmp_path):
    path = tmp_path / "sponsors.csv"
    path.write_text("Sponsor Login,Name,Email,Tier,Amount,Currency,Started At,Ended At,Status,Private\nold,Old,,Bronze,$5,USD,2026-04-01,2026-04-15,ended,false\n,Private,private@example.com,Gold,10,USD,2026-05-03,,active,true\n", encoding="utf-8")
    since = SyncState(source_project="github_sponsors_csv", source_entity_type="sponsor", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = GitHubSponsorsCsvAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["Private"]
    unit = result.units[0]
    assert unit.metadata["amount"] == 10.0
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["private"] is True
    assert unit.metadata["status"] == "active"
    assert GitHubSponsorsCsvAdapter(path=str(path)).ingest(entity_types=["repo"]).units == []
    assert get_adapter("github_sponsors_csv", path=str(path)).name == "github_sponsors_csv"
