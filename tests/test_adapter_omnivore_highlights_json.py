from __future__ import annotations

import json

from graph.adapters.omnivore_highlights_json import OmnivoreHighlightsJsonAdapter


def test_omnivore_highlights_json_ingests_article_highlights(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "article-1",
                    "title": "Article",
                    "url": "https://example.com/a",
                    "author": "Ada",
                    "labels": [{"name": "AI"}],
                    "highlights": [{"id": "h1", "text": "Important passage", "note": "Remember", "highlightedAt": "2024-01-01T00:00:00Z"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = OmnivoreHighlightsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Article"
    assert unit.metadata["highlight_text"] == "Important passage"
    assert unit.metadata["note"] == "Remember"
    assert unit.metadata["labels"] == ["ai"]


def test_omnivore_highlights_json_fallback_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(json.dumps([{"url": "https://e.test", "text": "Same"}]), encoding="utf-8")

    first = OmnivoreHighlightsJsonAdapter(path=str(export)).ingest().units[0].source_id
    second = OmnivoreHighlightsJsonAdapter(path=str(export)).ingest().units[0].source_id

    assert first == second
