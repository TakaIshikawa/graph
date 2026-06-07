from graph.store import summarize_source_sec_fetch_metadata


def test_sec_fetch_metadata_summary_counts_values_missing_unusual_and_samples():
    summary = summarize_source_sec_fetch_metadata(
        [
            {
                "source_id": "a",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-User": "?1",
            },
            {
                "source_id": "b",
                "metadata": {
                    "sec_fetch_site": "same-origin",
                    "sec_fetch_mode": "cors",
                    "sec_fetch_dest": "empty",
                    "sec_fetch_user": "?0",
                },
            },
            {
                "source_id": "c",
                "headers": {
                    "sec-fetch-site": "alien",
                    "sec-fetch-mode": "NESTED",
                    "sec-fetch-dest": "image",
                    "sec-fetch-user": "unexpected",
                },
            },
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_sec_fetch_metadata"] == 3
    assert summary["missing_metadata_count"] == 1
    assert summary["site_value_counts"] == {"alien": 1, "cross-site": 1, "same-origin": 1}
    assert summary["mode_value_counts"] == {"cors": 1, "navigate": 1, "nested": 1}
    assert summary["dest_value_counts"] == {"document": 1, "empty": 1, "image": 1}
    assert summary["user_value_counts"] == {"?0": 1, "?1": 1, "unexpected": 1}
    assert summary["cross_site_navigation_count"] == 1
    assert summary["unusual_value_count"] == 3
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
