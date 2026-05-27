from __future__ import annotations

from graph.adapters.pocket_articles_csv import PocketArticlesCsvAdapter


def test_pocket_articles_csv_ingests_article_rows(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,tags,excerpt,time added,time read,favorite,status\n"
        "Example,https://example.com,\"Read,Web\",Summary,2024-01-01,2024-01-02,1,archive\n",
        encoding="utf-8",
    )

    unit = PocketArticlesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Example"
    assert unit.metadata["url"] == "https://example.com"
    assert unit.metadata["excerpt"] == "Summary"
    assert unit.metadata["favorite"] is True
    assert unit.metadata["archived"] is True
    assert unit.tags == ["read", "web"]


def test_pocket_articles_csv_duplicate_urls_stable_order(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text("title,url\nB,https://example.com\nA,https://example.com\n", encoding="utf-8")

    first = PocketArticlesCsvAdapter(path=str(export)).ingest().units
    second = PocketArticlesCsvAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert len({unit.source_id for unit in first}) == 2
