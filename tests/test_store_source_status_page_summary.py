from graph.store import summarize_source_status_pages


def test_status_page_summary_counts_links_and_incident_history():
    summary = summarize_source_status_pages(
        [
            {"source_id": "a", "url": "https://vendor.statuspage.io", "content": "Incident history and uptime."},
            {"source_id": "b", "content": "Degraded performance and scheduled maintenance notice."},
            {"source_id": "c", "content": "Product docs."},
        ]
    )

    assert summary["sources_with_status_page_hints"] == 2
    assert summary["status_page_link_count"] == 1
    assert summary["incident_history_hint_count"] == 1
    assert summary["cue_counts"]["uptime"] == 1
    assert summary["cue_counts"]["degraded_performance"] == 1
    assert summary["cue_counts"]["maintenance"] == 1
    assert summary["samples"][0]["source_id"] == "a"
