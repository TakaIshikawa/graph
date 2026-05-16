from __future__ import annotations

from dataclasses import dataclass

from graph.rag.chunk_boundary_audit import audit_chunk_boundaries


@dataclass
class Result:
    id: str
    text: str


def test_chunk_boundary_audit_detects_boundary_and_delimiter_issues():
    report = audit_chunk_boundaries(
        [
            {"id": "a", "snippet": "...continued from previous sentence"},
            {"id": "b", "content": "This fragment trails because"},
            {"id": "c", "text": "The method uses (retrieval augmented generation"},
            {"id": "d", "text": "Tiny"},
        ]
    )

    assert [(issue["result_id"], issue["issue_type"], issue["severity"]) for issue in report["issues"]] == [
        ("c", "unmatched-delimiter", "high"),
        ("a", "leading-continuation", "medium"),
        ("b", "trailing-continuation", "medium"),
        ("b", "very-short-fragment", "low"),
        ("d", "very-short-fragment", "low"),
    ]
    assert report["summary"]["issue_type_counts"] == {
        "leading-continuation": 1,
        "trailing-continuation": 1,
        "unmatched-delimiter": 1,
        "very-short-fragment": 2,
    }


def test_chunk_boundary_audit_accepts_objects_tuples_and_metadata():
    report = audit_chunk_boundaries(
        [
            (Result("object", "however this starts in the middle"), 0.5),
            {"id": "meta", "metadata": {"snippet": "Quoted \"fragment"}},
        ]
    )

    assert [(issue["result_id"], issue["issue_type"]) for issue in report["issues"]] == [
        ("meta", "unmatched-delimiter"),
        ("object", "leading-continuation"),
        ("meta", "very-short-fragment"),
        ("object", "very-short-fragment"),
    ]


def test_chunk_boundary_audit_handles_empty_input():
    assert audit_chunk_boundaries([]) == {
        "issue_count": 0,
        "issues": [],
        "summary": {"result_count": 0, "issue_type_counts": {}},
    }
