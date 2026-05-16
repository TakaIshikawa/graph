from __future__ import annotations

from dataclasses import dataclass

from graph.rag.source_independence import analyze_source_independence


@dataclass
class Result:
    id: str
    text: str
    metadata: dict


def test_source_independence_groups_shared_source_signals():
    report = analyze_source_independence(
        [
            {"id": "a", "url": "https://www.example.com/a", "canonical_url": "https://example.com/story", "source_id": "wire-1"},
            {"id": "b", "url": "https://example.com/b", "canonical_url": "https://example.com/story", "source_id": "wire-1"},
            {"id": "c", "url": "https://other.test/c"},
        ]
    )

    assert report["dependent_result_ids"] == ["a", "b"]
    assert {group["reason"] for group in report["groups"]} == {"canonical_url", "domain", "source_id"}
    assert report["independence_score"] == 0.333333


def test_source_independence_accepts_objects_tuples_and_fingerprints():
    report = analyze_source_independence(
        [
            (Result("a", "alpha beta gamma delta epsilon", {"author": "Ada", "title": "Launch"}), 0.9),
            Result("b", "alpha beta gamma delta zeta", {"author": "Ada", "title": "Launch"}),
        ]
    )

    assert report["dependent_result_ids"] == ["a", "b"]
    assert [group["reason"] for group in report["groups"]] == ["author_title", "fingerprint"]
