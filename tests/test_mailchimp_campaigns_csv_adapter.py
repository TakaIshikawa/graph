from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.mailchimp_campaigns_csv import MailchimpCampaignsCsvAdapter
from graph.types.models import SyncState


def test_mailchimp_campaigns_csv_ingests_common_headers_and_metrics(tmp_path):
    export = tmp_path / "campaigns.csv"
    export.write_text(
        "Campaign ID,Title,Subject,Status,Send Time,List,Emails Sent,Opens,Clicks,Unsubscribes,Archive URL\n"
        "abc,April Update,News,Sent,2026-04-01 08:30:00,Main,1000,400,75,2,https://mailchi.mp/archive\n",
        encoding="utf-8",
    )

    unit = MailchimpCampaignsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "mailchimp_campaigns_csv"
    assert unit.source_entity_type == "campaign"
    assert unit.source_id == "mailchimp_campaigns_csv:abc"
    assert unit.metadata["emails_sent"] == 1000
    assert unit.metadata["opens"] == 400
    assert unit.metadata["clicks"] == 75
    assert unit.metadata["unsubscribes"] == 2
    assert "mailchimp" in unit.tags
    assert "Clicks: 75" in unit.content


def test_mailchimp_campaigns_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "campaigns.csv"
    export.write_text(
        "ID,Name,Email Subject,Status,Sent At,Audience,Recipients,Total Opens,Total Clicks\n"
        "new,New Campaign,New,Sent,2026-04-03,Customers,20,10,5\n"
        "old,Old Campaign,Old,Sent,2026-04-01,Customers,10,4,1\n"
        ",,,,,,,,\n"
        ",Title Only,,,,,,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="mailchimp_campaigns_csv", source_entity_type="campaign", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = MailchimpCampaignsCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["campaign_id"] for unit in units] == ["new"]
    all_units = MailchimpCampaignsCsvAdapter(path=str(export)).ingest().units
    assert [unit.title for unit in all_units] == ["Old Campaign", "New Campaign", "Title Only"]
    assert MailchimpCampaignsCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
