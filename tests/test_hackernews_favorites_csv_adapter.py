from __future__ import annotations

from graph.adapters.hackernews_favorites_csv import HackerNewsFavoritesCsvAdapter


def test_hackernews_favorites_csv_parses_articles_and_comments(tmp_path):
    path = tmp_path / "favorites.csv"
    path.write_text(
        "item_id,title,url,author,created_at,type,text\n"
        "1,Article,https://example.com,pg,2026-01-01T00:00:00Z,story,\n"
        "2,,https://news.ycombinator.com/item?id=2,alice,2026-01-02,comment,Interesting point\n",
        encoding="utf-8",
    )

    units = HackerNewsFavoritesCsvAdapter(path=str(path)).ingest().units

    assert [unit.title for unit in units] == ["Article", "Hacker News item 2"]
    assert units[0].metadata["url"] == "https://example.com"
    assert units[0].metadata["author"] == "pg"
    assert units[0].metadata["type"] == "story"
    assert units[1].metadata["created_at"] == "2026-01-02"
    assert "Interesting point" in units[1].content
