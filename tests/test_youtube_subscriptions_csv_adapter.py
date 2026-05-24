from __future__ import annotations

from graph.adapters.youtube_subscriptions_csv import YoutubeSubscriptionsCsvAdapter


def test_youtube_subscriptions_csv_handles_takeout_headers(tmp_path):
    export = tmp_path / "subscriptions.csv"
    export.write_text(
        "Channel Id,Channel Url,Channel Title,Subscribed Date\n"
        "UC123,https://www.youtube.com/channel/UC123,Example Channel,2026-05-01\n",
        encoding="utf-8",
    )

    unit = YoutubeSubscriptionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "youtube_subscriptions_csv:channel:UC123"
    assert unit.metadata["channel_id"] == "UC123"
    assert unit.metadata["title"] == "Example Channel"
    assert unit.metadata["url"] == "https://www.youtube.com/channel/UC123"
    assert unit.metadata["subscribed_at"] == "2026-05-01T00:00:00+00:00"


def test_youtube_subscriptions_csv_uses_url_as_identifier_without_channel_id(tmp_path):
    export = tmp_path / "subscriptions.csv"
    export.write_text("Channel Url,Channel Title\nhttps://youtube.com/@example,Example\n", encoding="utf-8")

    unit = YoutubeSubscriptionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id.startswith("youtube_subscriptions_csv:")
    assert unit.metadata["url"] == "https://youtube.com/@example"
