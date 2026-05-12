from __future__ import annotations

from graph.adapters.podcasts_opml import PodcastsOpmlAdapter
from graph.adapters.registry import get_adapter


def test_podcasts_opml_ingests_feeds_and_deduplicates(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text("""<opml><body><outline text="Tech"><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" htmlUrl="https://example.com" description="A show" ownerName="Ada" ownerEmail="ada@example.com" author="Graph Radio" language="en" imageUrl="https://example.com/art.png"/><outline text="Duplicate" type="rss" xmlUrl="https://example.com/feed.xml"/></outline></body></opml>""", encoding="utf-8")

    result = PodcastsOpmlAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Show"
    assert unit.metadata["xmlUrl"] == "https://example.com/feed.xml"
    assert unit.metadata["ownerName"] == "Ada"
    assert unit.metadata["ownerEmail"] == "ada@example.com"
    assert unit.metadata["author"] == "Graph Radio"
    assert unit.metadata["language"] == "en"
    assert unit.metadata["imageUrl"] == "https://example.com/art.png"
    assert unit.metadata["folder_path"] == ["Tech"]
    assert "Tech" in unit.tags
    assert "Graph Radio" in unit.tags
    assert "en" in unit.tags
    assert get_adapter("podcasts_opml", path=str(export)).name == "podcasts_opml"


def test_podcasts_opml_omits_blank_optional_metadata(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text("""<opml><body><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" ownerName="" ownerEmail=" " author="" language="" imageHref=" "/></body></opml>""", encoding="utf-8")

    unit = PodcastsOpmlAdapter(path=str(export)).ingest().units[0]

    assert "ownerName" not in unit.metadata
    assert "ownerEmail" not in unit.metadata
    assert "author" not in unit.metadata
    assert "language" not in unit.metadata
    assert "imageUrl" not in unit.metadata
