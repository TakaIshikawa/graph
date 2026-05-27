from __future__ import annotations

from graph.rag.evidence_temporal_ordering_audit import audit_evidence_temporal_ordering


class Evidence:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_evidence_temporal_ordering_classifies_chronological():
    assert audit_evidence_temporal_ordering([{"id": "a", "date": "2024-01-01"}, {"id": "b", "date": "2024-01-02"}])["ordering"] == "chronological"


def test_evidence_temporal_ordering_classifies_reverse_chronological():
    assert audit_evidence_temporal_ordering([{"date": "2024-01-02"}, {"date": "2024-01-01"}])["ordering"] == "reverse_chronological"


def test_evidence_temporal_ordering_reports_mixed_inversions():
    summary = audit_evidence_temporal_ordering([{"id": "a", "date": "2024-01-01"}, {"id": "b", "date": "2024-01-03"}, {"id": "c", "date": "2024-01-02"}])

    assert summary["ordering"] == "mixed"
    assert summary["inversions"] == [{"left_id": "b", "left_date": "2024-01-03", "right_id": "c", "right_date": "2024-01-02"}]


def test_evidence_temporal_ordering_handles_undated_and_object_metadata():
    summary = audit_evidence_temporal_ordering([Evidence(metadata={"published_at": "2024-01-01"}), {}])

    assert summary["dated_count"] == 1
    assert summary["undated_count"] == 1
    assert summary["earliest_date"] == "2024-01-01"
