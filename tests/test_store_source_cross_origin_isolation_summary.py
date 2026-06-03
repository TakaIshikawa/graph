from graph.store import summarize_source_cross_origin_isolation


def test_cross_origin_isolation_summary_combines_policies_and_missing_counts():
    summary = summarize_source_cross_origin_isolation(
        [
            {"source_id": "ready", "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin"},
            {"source_id": "weak", "response_headers": {"Cross-Origin-Opener-Policy": "same-origin-allow-popups", "Cross-Origin-Embedder-Policy": "credentialless"}},
            {"source_id": "missing", "headers": {"Cross-Origin-Resource-Policy": "cross-origin"}},
        ],
    )

    assert summary["total_sources"] == 3
    assert summary["isolated_candidate_count"] == 1
    assert summary["missing_policy_counts"] == {"coop": 1, "coep": 1, "corp": 1}
    assert summary["rows"] == [
        {"source_id": "missing", "coop": "", "coep": "", "corp": "cross-origin", "isolated_candidate": False},
        {"source_id": "ready", "coop": "same-origin", "coep": "require-corp", "corp": "same-origin", "isolated_candidate": True},
        {"source_id": "weak", "coop": "same-origin-allow-popups", "coep": "credentialless", "corp": "", "isolated_candidate": False},
    ]
