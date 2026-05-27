from __future__ import annotations

from graph.rag.evidence_version_conflict import analyze_evidence_version_conflicts


class Evidence:
    def __init__(self, evidence_id: str, text: str, metadata: dict | None = None):
        self.id = evidence_id
        self.text = text
        self.metadata = metadata or {}


def test_analyze_evidence_version_conflicts_groups_semantic_versions():
    result = analyze_evidence_version_conflicts(
        [
            {"id": "e1", "text": "Package behavior changed in version 1.2.0."},
            {"id": "e2", "text": "The package docs for v1.3.0 describe a new default."},
            {"id": "e3", "text": "No version listed."},
        ]
    )

    assert result["total_evidence"] == 3
    assert result["versioned_evidence_count"] == 2
    assert result["version_groups"] == [
        {"version": "1.2.0", "evidence_ids": ["e1"], "count": 1},
        {"version": "1.3.0", "evidence_ids": ["e2"], "count": 1},
    ]
    assert result["conflict_count"] == 1
    assert result["conflict_flags"] == [
        {"type": "multiple_versions", "versions": ["1.2.0", "1.3.0"], "evidence_ids": ["e1", "e2"]}
    ]


def test_analyze_evidence_version_conflicts_uses_metadata_versions():
    result = analyze_evidence_version_conflicts(
        [
            Evidence("e1", "Release notes", {"version": "v2.0.0"}),
            Evidence("e2", "API guide", {"api_version": "2.1.0"}),
        ]
    )

    assert result["version_groups"] == [
        {"version": "2.0.0", "evidence_ids": ["e1"], "count": 1},
        {"version": "2.1.0", "evidence_ids": ["e2"], "count": 1},
    ]
    assert result["conflict_count"] == 1


def test_analyze_evidence_version_conflicts_ignores_missing_versions():
    result = analyze_evidence_version_conflicts([{"id": "e1", "text": "No release marker here."}])

    assert result == {
        "total_evidence": 1,
        "versioned_evidence_count": 0,
        "version_groups": [],
        "conflict_count": 0,
        "conflict_flags": [],
    }


def test_analyze_evidence_version_conflicts_does_not_flag_same_version():
    result = analyze_evidence_version_conflicts(
        [
            {"id": "e2", "text": "Docs apply to v3.4.5."},
            {"id": "e1", "metadata": {"version": "3.4.5"}, "text": "Same release."},
        ]
    )

    assert result["version_groups"] == [{"version": "3.4.5", "evidence_ids": ["e1", "e2"], "count": 2}]
    assert result["conflict_count"] == 0
    assert result["conflict_flags"] == []
