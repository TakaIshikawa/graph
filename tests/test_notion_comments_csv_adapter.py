from __future__ import annotations

from graph.adapters.notion_comments_csv import NotionCommentsCsvAdapter
from graph.adapters.registry import get_adapter


def test_notion_comments_csv_ingests_comments_and_ids(tmp_path):
    path = tmp_path / "comments.csv"
    path.write_text("Comment ID,Page Title,Comment,Author,Created Time,Resolved,URL\nc1,Roadmap,Looks good,Ada,2026-05-01T10:00:00Z,true,https://notion.test/p\n", encoding="utf-8")

    unit = NotionCommentsCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "notion_comments_csv"
    assert unit.source_id == "notion_comments_csv:c1"
    assert unit.source_entity_type == "comment"
    assert unit.metadata["page_title"] == "Roadmap"
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["resolved"] is True
    assert unit.metadata["source_file"] == "comments.csv"
    assert isinstance(get_adapter("notion_comments_csv"), NotionCommentsCsvAdapter)
