from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.linkedin_saved_posts_csv import LinkedInSavedPostsCsvAdapter
from graph.types.models import SyncState


def test_linkedin_saved_posts_csv_ingests_posts_and_engagement_counts(tmp_path):
    export = tmp_path / "saved-posts.csv"
    export.write_text(
        "Saved Date,Author,Author Profile URL,Post URL,Text,Reaction Count,Comment Count,Tags\n"
        "2026-05-01,Jane Doe,https://linkedin.com/in/jane,https://linkedin.com/posts/1,Useful post,1,234,\"career, ai\"\n"
        "2026-05-02,,,,Text-only post,,,\n",
        encoding="utf-8",
    )

    result = LinkedInSavedPostsCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["saved_post", "saved_post"]
    unit = result.units[0]
    assert unit.metadata["author_profile_url"] == "https://linkedin.com/in/jane"
    assert unit.metadata["saved_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["reaction_count"] == 1
    assert unit.metadata["comment_count"] == 234
    assert unit.metadata["tags"] == ["career", "ai"]
    assert "Reactions: 1" in unit.content


def test_linkedin_saved_posts_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "saved-posts.csv"
    export.write_text(
        "Saved Date,Post URL,Text\n"
        "2026-05-01,https://linkedin.com/posts/1,Old\n"
        "2026-05-03,https://linkedin.com/posts/2,New\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="linkedin_saved_posts_csv", source_entity_type="saved_post", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = LinkedInSavedPostsCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.metadata["text"] for unit in result.units] == ["New"]
    assert LinkedInSavedPostsCsvAdapter(path=str(export)).ingest(entity_types=["post"]).units == []
