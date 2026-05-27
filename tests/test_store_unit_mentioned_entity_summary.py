from __future__ import annotations

from graph.store.unit_mentioned_entity_summary import summarize_unit_mentioned_entities


def test_unit_mentioned_entities_counts_mixed_entity_shapes():
    summary = summarize_unit_mentioned_entities(
        [
            {"source_project": "notes", "metadata": {"people": ["Ada", {"name": "Grace"}], "topics": "AI, Math"}},
            {"source_project": "notes", "entities": [{"text": "Ada"}, {"id": "ACM"}]},
            {"source_project": "notes", "metadata": {}},
            {"source_project": "web", "metadata": {"organizations": "ACM"}},
        ]
    )

    rows = {row["source"]: row for row in summary["rows"]}
    assert rows["notes"]["unit_count"] == 3
    assert rows["notes"]["entity_unit_count"] == 2
    assert rows["notes"]["total_entity_mentions"] == 6
    assert rows["notes"]["distinct_entity_count"] == 5
    assert rows["notes"]["average_entities_per_unit"] == 2.0
    assert rows["notes"]["top_entities"][0] == {"entity": "Ada", "count": 2}


def test_unit_mentioned_entities_orders_sources_and_top_entities():
    summary = summarize_unit_mentioned_entities(
        [{"source_project": "b", "metadata": {"topics": "Beta, Alpha"}}, {"source_project": "a", "metadata": {"topics": ""}}]
    )

    assert [row["source"] for row in summary["rows"]] == ["a", "b"]
    assert summary["rows"][1]["top_entities"] == [{"entity": "Alpha", "count": 1}, {"entity": "Beta", "count": 1}]
