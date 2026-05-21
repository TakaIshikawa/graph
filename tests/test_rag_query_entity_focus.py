from __future__ import annotations

from graph.rag import extract_query_entity_focus


def test_query_entity_focus_extracts_common_entity_shapes():
    rows = extract_query_entity_focus('Compare "Project Atlas" with Ada Lovelace at @OpenAI for example.com #RAG')

    assert [row["type"] for row in rows] == ["quoted_phrase", "capitalized_name", "handle", "url_or_domain", "hashtag"]
    assert rows[0]["text"] == "project atlas"
    assert rows[1]["original_text"] == "Ada Lovelace"
    assert rows[3]["text"] == "example.com"


def test_query_entity_focus_returns_empty_and_collapses_duplicates():
    assert extract_query_entity_focus("how should I plan this?") == []

    rows = extract_query_entity_focus('"Project Atlas" and "Project Atlas"')
    assert len(rows) == 1
    assert rows[0]["position"] == 0
