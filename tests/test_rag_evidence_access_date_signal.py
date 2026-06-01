from graph.rag.evidence_access_date_signal import analyze_evidence_access_date_signals


class Evidence:
    def __init__(self, id, title, content):
        self.id = id
        self.title = title
        self.content = content


def test_recognizes_metadata_and_content_access_dates():
    summary = analyze_evidence_access_date_signals(
        [
            {"id": "a", "metadata": {"access_date": "2026-01-02"}},
            Evidence("b", "Doc B", "Retrieved on 2026-01-03 by the crawler."),
            {"id": "c", "title": "Doc C", "text": "No access marker."},
        ]
    )

    assert summary["total_evidence"] == 3
    assert summary["with_access_date"] == 2
    assert summary["missing_access_date"] == 1
    assert summary["samples"] == [{"index": 2, "source_id": "c", "title": "Doc C"}]


def test_empty_evidence_returns_zero_counts():
    assert analyze_evidence_access_date_signals([]) == {
        "total_evidence": 0,
        "with_access_date": 0,
        "missing_access_date": 0,
        "samples": [],
    }
