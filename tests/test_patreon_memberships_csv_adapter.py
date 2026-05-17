from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.patreon_memberships_csv import PatreonMembershipsCsvAdapter
from graph.types.models import SyncState


def test_patreon_memberships_csv_ingests_active_member_dates_amounts_and_tags(tmp_path):
    export = tmp_path / "patreon_members.csv"
    export.write_text(
        "Member ID,Full Name,Email,Tier,Patron Status,Pledge Amount,Lifetime Amount,Currency,Join Date,Last Charge Date,Last Charge Status,Next Charge Date,Address Country,Note\n"
        "m_1,Ada Lovelace,ada@example.com,Supporter,active,$5.00,$120.50,USD,2025-01-02,2026-05-01,paid,2026-06-01,US,Founding patron\n",
        encoding="utf-8",
    )

    result = PatreonMembershipsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "patreon_memberships_csv"
    assert unit.source_id == "patreon_memberships_csv:m_1"
    assert unit.source_entity_type == "membership"
    assert unit.metadata["full_name"] == "Ada Lovelace"
    assert unit.metadata["email"] == "ada@example.com"
    assert unit.metadata["tier"] == "Supporter"
    assert unit.metadata["patron_status"] == "active"
    assert unit.metadata["pledge_amount"] == 5.0
    assert unit.metadata["lifetime_amount"] == 120.5
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["join_date"] == "2025-01-02T00:00:00+00:00"
    assert unit.metadata["last_charge_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["last_charge_status"] == "paid"
    assert unit.metadata["next_charge_date"] == "2026-06-01T00:00:00+00:00"
    assert unit.metadata["address_country"] == "US"
    assert unit.metadata["note"] == "Founding patron"
    assert unit.created_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert {"patreon", "membership", "Supporter", "active", "paid", "US"}.issubset(set(unit.tags))
    assert "Pledge amount: 5.0 USD" in unit.content


def test_patreon_memberships_csv_directory_aliases_declined_since_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Patron Id,Patron Name,Patron Email,Tier Title,Status,Current Pledge,Lifetime Support,Currency,Patron Since,Last Payment Date,Last Payment Status,Country,Notes\n"
        "old-1,Old Member,old@example.com,Basic,active,3,9,USD,2025-01-01,2026-04-01,paid,CA,Old note\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "User ID,Name,Email Address,Membership Tier,Membership Status,Amount,Total Pledged,ISO Currency Code,Joined At,Last Charged At,Charge Status,Next Payment Date,Shipping Country,Patron Note\n"
        "new-1,Grace Hopper,grace@example.com,VIP,declined,10,50,USD,2025-02-01,2026-05-03,declined,2026-06-03,GB,Card failed\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not csv", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    sync = SyncState(source_project="patreon_memberships_csv", source_entity_type="membership", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    adapter = PatreonMembershipsCsvAdapter(path=str(tmp_path))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)
    all_units = adapter.ingest().units

    assert [unit.metadata["full_name"] for unit in first.units] == ["Grace Hopper"]
    assert first.units[0].metadata["patron_status"] == "declined"
    assert first.units[0].metadata["last_charge_status"] == "declined"
    assert first.units[0].metadata["pledge_amount"] == 10.0
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [unit.metadata["full_name"] for unit in all_units] == ["Old Member", "Grace Hopper"]
    assert adapter.ingest(entity_types=["patron"]).units == []


def test_patreon_memberships_csv_fallback_id_and_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text(
        "Full Name,Email,Tier,Patron Status,Pledge Amount,Note\n"
        ",,,,,\n"
        "Sparse Patron,sparse@example.com,,active,2.50,\n",
        encoding="utf-8",
    )

    first = PatreonMembershipsCsvAdapter(path=str(export)).ingest().units[0]
    second = PatreonMembershipsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("patreon_memberships_csv:")
    assert first.metadata["full_name"] == "Sparse Patron"
    assert first.metadata["pledge_amount"] == 2.5
    assert "tier" not in first.metadata
