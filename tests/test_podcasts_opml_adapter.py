from __future__ import annotations

from graph.adapters.podcasts_opml import PodcastsOpmlAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource


def test_podcasts_opml_ingests_feeds_and_deduplicates(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text("""<opml><body><outline text="Tech"><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" htmlUrl="https://example.com" description="A show" ownerName="Ada" ownerEmail="ada@example.com" author="Graph Radio" language="en" imageUrl="https://example.com/art.png"/><outline text="Duplicate" type="rss" xmlUrl="https://example.com/feed.xml"/></outline></body></opml>""", encoding="utf-8")

    result = PodcastsOpmlAdapter(path=str(export)).ingest()

    unit = next(unit for unit in result.units if unit.source_entity_type == "podcast")
    assert unit.title == "Show"
    assert unit.metadata["xmlUrl"] == "https://example.com/feed.xml"
    assert unit.metadata["ownerName"] == "Ada"
    assert unit.metadata["ownerEmail"] == "ada@example.com"
    assert unit.metadata["author"] == "Graph Radio"
    assert unit.metadata["language"] == "en"
    assert unit.metadata["imageUrl"] == "https://example.com/art.png"
    assert unit.metadata["folder_path"] == ["Tech"]
    assert unit.metadata["categories"] == ["Tech"]
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


def test_podcasts_opml_emits_category_units_and_edges(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text(
        """<opml><body><outline text="Tech"><outline text="AI"><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" category="Machine Learning; AI" genre="Technology"/></outline></outline><outline text="News"><outline text="Daily" type="rss" xmlUrl="https://example.com/daily.xml" genre="Technology"/></outline></body></opml>""",
        encoding="utf-8",
    )

    result = PodcastsOpmlAdapter(path=str(export)).ingest(entity_types=["podcast", "category"])
    second = PodcastsOpmlAdapter(path=str(export)).ingest(entity_types=["podcast", "category"])

    assert PodcastsOpmlAdapter(path=str(export)).entity_types == ["podcast", "category"]
    podcasts = [unit for unit in result.units if unit.source_entity_type == "podcast"]
    categories = sorted((unit for unit in result.units if unit.source_entity_type == "category"), key=lambda unit: unit.title)
    assert [unit.title for unit in categories] == ["AI", "Machine Learning", "News", "Tech", "Technology"]
    technology = next(unit for unit in categories if unit.title == "Technology")
    assert technology.metadata["podcast_count"] == 2
    assert technology.metadata["podcast_source_ids"] == sorted(podcast.source_id for podcast in podcasts)
    assert [unit.source_id for unit in categories] == [
        unit.source_id for unit in sorted((u for u in second.units if u.source_entity_type == "category"), key=lambda u: u.title)
    ]
    assert len(result.edges) == 6
    assert {edge.relation for edge in result.edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.from_unit_id for edge in result.edges} == {podcast.source_id for podcast in podcasts}
    assert {edge.to_unit_id for edge in result.edges} == {category.source_id for category in categories}
    assert [edge.id for edge in result.edges] == [edge.id for edge in second.edges]


def test_podcasts_opml_category_filtering_without_edges(tmp_path):
    export = tmp_path / "podcasts.opml"
    export.write_text(
        """<opml><body><outline text="Tech"><outline text="Show" type="rss" xmlUrl="https://example.com/feed.xml" category="Machine Learning"/></outline></body></opml>""",
        encoding="utf-8",
    )

    result = PodcastsOpmlAdapter(path=str(export)).ingest(entity_types=["category"])

    assert {unit.source_entity_type for unit in result.units} == {"category"}
    assert {unit.title for unit in result.units} == {"Tech", "Machine Learning"}
    assert result.edges == []
