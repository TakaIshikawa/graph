from __future__ import annotations

from graph.adapters.podcasts_opml import PodcastsOpmlAdapter
from graph.adapters.registry import get_adapter


def test_podcasts_opml_ingests_feeds_and_deduplicates(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text("""<opml><body><outline text="Tech"><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" htmlUrl="https://example.com" description="A show"/><outline text="Duplicate" type="rss" xmlUrl="https://example.com/feed.xml"/></outline></body></opml>""", encoding="utf-8")

    result = PodcastsOpmlAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Show"
    assert unit.metadata["xmlUrl"] == "https://example.com/feed.xml"
    assert unit.metadata["folder_path"] == ["Tech"]
    assert "Tech" in unit.tags
    assert get_adapter("podcasts_opml", path=str(export)).name == "podcasts_opml"
