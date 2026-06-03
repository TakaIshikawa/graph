from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.linkedin_connections_csv import LinkedInConnectionsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_linkedin_connections_csv_handles_preamble_sparse_rows_filters_and_registry(tmp_path):
    path = tmp_path / "Connections.csv"
    path.write_text("\ufeffExported from LinkedIn\n\nFirst Name,Last Name,Email Address,Company,Position,Connected On,Profile URL,Notes\nOld,Person,old@example.com,Old Co,Dev,04/01/2026,,\nNew,Person,,New Co,Lead,05/03/2026,https://linkedin/in/new,Met\n,,,,,,,\n", encoding="utf-8")
    since = SyncState(source_project="linkedin_connections_csv", source_entity_type="connection", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = LinkedInConnectionsCsvAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New Person"]
    unit = result.units[0]
    assert unit.metadata["company"] == "New Co"
    assert unit.metadata["position"] == "Lead"
    assert unit.metadata["profile_url"] == "https://linkedin/in/new"
    assert unit.source_id == LinkedInConnectionsCsvAdapter(path=str(path)).ingest().units[1].source_id
    assert LinkedInConnectionsCsvAdapter(path=str(path)).ingest(entity_types=["post"]).units == []
    assert get_adapter("linkedin_connections_csv", path=str(path)).name == "linkedin_connections_csv"
