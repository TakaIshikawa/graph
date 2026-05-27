from __future__ import annotations

from types import SimpleNamespace

from graph.store.relation_metadata_key_frequency_summary import summarize_relation_metadata_key_frequency


def test_relation_metadata_key_frequency_counts_nested_dotted_paths():
    summary = summarize_relation_metadata_key_frequency(
        [
            {"type": "cites", "metadata": {"Evidence": {"url": "https://example.com", "Quote": "x"}, "confidence": 0.9}},
            {"type": "cites", "metadata": {"evidence": {"url": "https://example.org"}, "source": "notes"}},
        ]
    )

    assert summary["total_edges"] == 2
    assert summary["relation_summaries"] == [
        {
            "relation_type": "cites",
            "edge_count": 2,
            "metadata_key_count": 5,
            "keys": [
                {"key": "evidence", "count": 2},
                {"key": "evidence.url", "count": 2},
                {"key": "confidence", "count": 1},
                {"key": "evidence.quote", "count": 1},
                {"key": "source", "count": 1},
            ],
        }
    ]


def test_relation_metadata_key_frequency_groups_by_relation_type_and_sorts_rows():
    summary = summarize_relation_metadata_key_frequency(
        [
            SimpleNamespace(relation_type="mentions", metadata={"source": "crm"}),
            {"type": "cites", "metadata": {"source": "notes"}},
            {"type": "cites", "metadata": {"source": "web", "confidence": 1}},
            {"metadata": {"reviewed": True}},
        ]
    )

    assert [row["relation_type"] for row in summary["relation_summaries"]] == ["cites", "mentions", "unknown"]
    assert summary["relation_summaries"][0]["keys"] == [{"key": "source", "count": 2}, {"key": "confidence", "count": 1}]
    assert summary["top_keys"] == [
        {"key": "source", "count": 3},
        {"key": "confidence", "count": 1},
        {"key": "reviewed", "count": 1},
    ]


def test_relation_metadata_key_frequency_handles_empty_metadata():
    summary = summarize_relation_metadata_key_frequency([{"type": "links"}, {"type": "links", "metadata": {}}])

    assert summary == {
        "total_edges": 2,
        "relation_summaries": [{"relation_type": "links", "edge_count": 2, "metadata_key_count": 0, "keys": []}],
        "top_keys": [],
    }
