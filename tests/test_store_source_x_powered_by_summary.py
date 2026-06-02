from graph.store import summarize_source_x_powered_by


def test_x_powered_by_summary_groups_common_technologies_and_export():
    summary = summarize_source_x_powered_by(
        [
            {"id": "a", "headers": {"X-Powered-By": "PHP/8.2"}},
            {"id": "b", "metadata": {"response_headers": {"x_powered_by": "Express"}}},
            {"id": "c", "response_headers": {"X-POWERED-BY": "ASP.NET"}},
            {"id": "d"},
        ]
    )

    assert summary["sources_with_x_powered_by"] == 3
    assert summary["missing_x_powered_by_count"] == 1
    assert summary["technology_counts"] == {"ASP.NET": 1, "Express": 1, "PHP": 1}
    assert summary["value_counts"] == {"ASP.NET": 1, "Express": 1, "PHP/8.2": 1}
