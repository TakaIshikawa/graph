from graph.store import summarize_source_origin_agent_clusters


def test_origin_agent_cluster_summary_normalizes_values():
    summary = summarize_source_origin_agent_clusters(
        [
            {"source_id": "a", "Origin-Agent-Cluster": "?1"},
            {"source_id": "b", "metadata": {"headers": {"origin-agent-cluster": "?0"}}},
            {"source_id": "c", "response_headers": {"Origin-Agent-Cluster": "true"}},
            {"source_id": "d", "headers": {"origin_agent_cluster": "maybe"}},
            {"source_id": "e"},
        ]
    )

    assert summary["sources_with_header"] == 4
    assert summary["enabled_count"] == 2
    assert summary["disabled_count"] == 1
    assert summary["unknown_value_count"] == 1
    assert summary["missing_header_count"] == 1
    assert summary["value_counts"] == {"?0": 1, "?1": 1, "maybe": 1, "true": 1}
    assert [sample["source_id"] for sample in summary["samples"]] == ["b", "d"]
