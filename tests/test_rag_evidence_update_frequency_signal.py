from graph.rag.evidence_update_frequency_signal import analyze_evidence_update_frequency_signals


class Evidence:
    def __init__(self, id, content):
        self.id = id
        self.content = content


def test_groups_update_frequency_phrases_from_multiple_shapes():
    summary = analyze_evidence_update_frequency_signals(
        [
            "Updated daily with new filings.",
            {"id": "w", "metadata": {"note": "updated weekly"}},
            Evidence("d", "This endpoint is deprecated."),
            {"id": "a", "text": "Archived and no longer maintained."},
        ]
    )

    assert summary["cadence_counts"] == {"archived": 1, "daily": 1, "deprecated": 1, "weekly": 1}
    assert summary["stale_signal_count"] == 2
    assert summary["examples"] == [
        {"source_id": "a", "cadence": "archived"},
        {"source_id": "result-1", "cadence": "daily"},
        {"source_id": "d", "cadence": "deprecated"},
        {"source_id": "w", "cadence": "weekly"},
    ]
