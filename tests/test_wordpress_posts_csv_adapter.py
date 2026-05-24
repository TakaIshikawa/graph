from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.wordpress_posts_csv import WordPressPostsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_wordpress_posts_csv_ingests_posts_and_splits_taxonomies(tmp_path):
    export = tmp_path / "posts.csv"
    export.write_text(
        "ID,Title,Slug,Status,Author,Date,Modified,URL,Categories,Tags,Excerpt,Content\n"
        "42,Launch,launch,publish,Ada,2026-04-01,2026-04-02,https://example.test/launch,\"News|Updates\",\"alpha,beta\",Short,Full body\n",
        encoding="utf-8",
    )

    unit = WordPressPostsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "wordpress_posts_csv"
    assert unit.source_entity_type == "post"
    assert unit.source_id == "wordpress_posts_csv:42"
    assert unit.metadata["categories"] == ["News", "Updates"]
    assert unit.metadata["tags"] == ["alpha", "beta"]
    assert unit.metadata["categories_raw"] == "News|Updates"
    assert unit.content == "Full body"
    assert unit.content_type == ContentType.ARTIFACT
    assert "News" in unit.tags
    assert "beta" in unit.tags


def test_wordpress_posts_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "posts.csv"
    export.write_text(
        "Post ID,Post Title,Post Slug,Post Status,Post Author,Post Date,Permalink,Category,Post Tags\n"
        "2,New,new,publish,Babbage,2026-04-03,https://example.test/new,Notes,one|two\n"
        "1,Old,old,draft,Ada,2026-04-01,https://example.test/old,Archive,old\n"
        ",,,,,,,,\n"
        ",Title Only,,,,,,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="wordpress_posts_csv", source_entity_type="post", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = WordPressPostsCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.title for unit in units] == ["New"]
    all_units = WordPressPostsCsvAdapter(path=str(export)).ingest().units
    assert [unit.title for unit in all_units] == ["Old", "New", "Title Only"]
    assert WordPressPostsCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
