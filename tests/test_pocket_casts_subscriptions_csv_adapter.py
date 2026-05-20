from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pocket_casts_subscriptions_csv import PocketCastsSubscriptionsCsvAdapter
from graph.types.models import SyncState


def test_pocket_casts_subscriptions_csv_ingests_subscription_metadata(tmp_path):
    export = tmp_path / "subscriptions.csv"
    export.write_text(
        "\n".join(
            [
                "Podcast Title,Author,Feed URL,Website URL,Description,Subscribed At,Categories,Episode Count,Last Published At",
                "Tech Show,Ada Lovelace,https://example.com/feed.xml,https://example.com,A good show,2025-01-02T03:04:05Z,Technology; News,42,2025-01-10T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )

    result = PocketCastsSubscriptionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "pocket_casts_subscriptions_csv"
    assert unit.source_entity_type == "podcast_subscription"
    assert unit.title == "Tech Show"
    assert unit.metadata["title"] == "Tech Show"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["feed_url"] == "https://example.com/feed.xml"
    assert unit.metadata["website_url"] == "https://example.com"
    assert unit.metadata["description"] == "A good show"
    assert unit.metadata["categories"] == ["Technology", "News"]
    assert unit.metadata["episode_count"] == 42
    assert unit.metadata["subscribed_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["last_published_at"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["source_file"] == "subscriptions.csv"
    assert unit.metadata["source_row"] == 1
    assert unit.metadata["row"]["Podcast Title"] == "Tech Show"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 10, tzinfo=timezone.utc)


def test_pocket_casts_subscriptions_csv_imports_rows_missing_feed_url_with_stable_title_id(tmp_path):
    export = tmp_path / "subscriptions.csv"
    export.write_text(
        "\n".join(
            [
                "Podcast Title,Author,Categories,Episode Count",
                "No Feed Show,Example Host,\"Comedy, Interviews\",12",
            ]
        ),
        encoding="utf-8",
    )

    adapter = PocketCastsSubscriptionsCsvAdapter(path=str(export))
    first = adapter.ingest().units[0]
    second = adapter.ingest().units[0]

    assert first.title == "No Feed Show"
    assert first.metadata["categories"] == ["Comedy", "Interviews"]
    assert first.metadata["episode_count"] == 12
    assert first.source_id == second.source_id
    assert first.source_id.startswith("pocket_casts_subscriptions_csv:")


def test_pocket_casts_subscriptions_csv_skips_blank_rows_and_filters_since_and_entity_type(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Podcast Title,Feed URL,Last Published At\nOld,https://example.com/old.xml,2025-01-01\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "Podcast Title,Feed URL,Last Published At\nNew,https://example.com/new.xml,2025-01-03\n,,\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    sync = SyncState(
        source_project="pocket_casts_subscriptions_csv",
        source_entity_type="podcast_subscription",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    adapter = PocketCastsSubscriptionsCsvAdapter(path=str(tmp_path))

    assert [unit.title for unit in adapter.ingest(since=sync).units] == ["New"]
    assert adapter.ingest(entity_types=["podcast_listen"]).units == []
