from graph.store.source_priority_header_summary import summarize_source_priority_headers


def test_source_priority_header_summary_counts_urgency_and_incremental():
    summary = summarize_source_priority_headers(
        [
            {"source_id": "a", "Priority": "u=0, i"},
            {"source_id": "b", "metadata": {"headers": {"priority": " u = 3 , i=true "}}},
            {"source_id": "c", "response_headers": {"Priority": "u=7, i=false"}},
            {"source_id": "d", "metadata": {"priority_header": "u=3,i=1"}},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["urgency_counts"] == {"0": 1, "3": 2, "7": 1}
    assert summary["incremental_true_count"] == 3
    assert summary["malformed_count"] == 0
    assert summary["missing_header_count"] == 0


def test_source_priority_header_summary_flags_malformed_and_missing_headers():
    summary = summarize_source_priority_headers(
        [
            {"source_id": "bad-urgency", "headers": {"Priority": "u=9"}},
            {"source_id": "bad-flag", "metadata": {"response_headers": {"priority": "i=maybe"}}},
            {"source_id": "unknown", "priority": "u=1, weight=10"},
            {"source_id": "missing"},
        ],
        sample_limit=3,
    )

    assert summary["urgency_counts"] == {"1": 1}
    assert summary["incremental_true_count"] == 0
    assert summary["malformed_count"] == 3
    assert summary["missing_header_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["bad-flag", "bad-urgency", "unknown"]


def test_source_priority_header_summary_bounds_examples():
    summary = summarize_source_priority_headers(
        [
            {"source_id": "a", "Priority": "i"},
            {"source_id": "b", "Priority": "u=2"},
        ],
        sample_limit=1,
    )

    assert summary["samples"] == [
        {
            "source_id": "a",
            "field": "Priority",
            "priority": "i",
            "urgency": None,
            "incremental": True,
            "malformed": False,
        }
    ]
