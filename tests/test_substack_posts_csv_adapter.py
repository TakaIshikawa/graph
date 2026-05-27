from __future__ import annotations

from graph.adapters.substack_posts_csv import SubstackPostsCsvAdapter


def test_substack_posts_csv_ingests_post_metrics(tmp_path):
    export = tmp_path / "posts.csv"
    export.write_text(
        "title,subtitle,post_date,canonical_url,audience,type,likes,comments,body\nLaunch,Hello,2024-01-01T00:00:00Z,https://example.substack.com/p/launch,everyone,newsletter,10,2,Body text\n",
        encoding="utf-8",
    )

    unit = SubstackPostsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Launch"
    assert unit.metadata["subtitle"] == "Hello"
    assert unit.metadata["canonical_url"] == "https://example.substack.com/p/launch"
    assert unit.metadata["audience"] == "everyone"
    assert unit.metadata["likes"] == 10
    assert unit.metadata["comments"] == 2
    assert "Body text" in unit.content


def test_substack_posts_csv_allows_missing_body_and_stable_fallback(tmp_path):
    export = tmp_path / "posts.csv"
    export.write_text("title,subtitle,post_date\nLaunch,Hello,2024-01-01\n", encoding="utf-8")

    first = SubstackPostsCsvAdapter(path=str(export)).ingest().units[0]
    second = SubstackPostsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert "Hello" in first.content
