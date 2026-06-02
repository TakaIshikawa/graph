from graph.store.source_feature_policy_summary import summarize_source_feature_policies


def test_feature_policy_summary_counts_features_disabled_and_malformed_directives():
    summary = summarize_source_feature_policies(
        [
            {"source_id": "a", "Feature-Policy": "geolocation 'none'; camera *; payment"},
            {"source_id": "b", "metadata": {"headers": {"FEATURE_POLICY": 'microphone "self"; fullscreen none'}}},
            {"source_id": "c"},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_feature_policy"] == 2
    assert summary["feature_counts"] == {"camera": 1, "fullscreen": 1, "geolocation": 1, "microphone": 1}
    assert summary["disabled_feature_counts"] == {"fullscreen": 1, "geolocation": 1}
    assert summary["malformed_directive_count"] == 1
    assert summary["missing_feature_policy_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "feature": "geolocation", "allowlist": "'none'"},
        {"source_id": "a", "feature": "camera", "allowlist": "*"},
    ]
