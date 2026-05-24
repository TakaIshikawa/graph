from __future__ import annotations

from dataclasses import dataclass

from graph.rag.evidence_peer_review_status import classify_evidence_peer_review_status


@dataclass
class EvidenceStub:
    id: str
    title: str = ""
    source: str = ""
    venue: str = ""
    metadata: dict | None = None


def test_classifies_common_peer_review_status_cues():
    report = classify_evidence_peer_review_status(
        [
            {"id": "journal", "title": "Journal of Retrieval study", "venue": "Nature"},
            {"id": "preprint", "title": "Scaling retrieval augmented generation", "url": "https://arxiv.org/abs/1234"},
            {"id": "report", "source_type": "technical report", "publisher": "Research Institute"},
            {"id": "news", "source": "Reuters", "title": "News: retrieval tools expand"},
            {"id": "docs", "title": "API reference", "url": "https://docs.example.com/rag"},
            {"id": "unknown", "title": "Meeting notes"},
        ]
    )

    assert report["status_counts"] == {
        "peer_reviewed": 1,
        "preprint": 1,
        "report": 1,
        "news_or_blog": 1,
        "documentation": 1,
        "unknown": 1,
    }
    assert [row["peer_review_status"] for row in report["per_evidence"]] == [
        "peer_reviewed",
        "preprint",
        "report",
        "news_or_blog",
        "documentation",
        "unknown",
    ]
    assert report["per_evidence"][0]["reasons"] == ["source_venue", "title_text"]
    assert report["per_evidence"][1]["reasons"] == ["domain", "source_venue"]
    assert report["per_evidence"][5]["reasons"] == ["insufficient_peer_review_signals"]


def test_metadata_status_takes_precedence_over_conflicting_text_cues():
    report = classify_evidence_peer_review_status(
        [
            {
                "id": "accepted",
                "publication_type": "journal_article",
                "title": "Preprint version on arXiv",
                "url": "https://arxiv.org/abs/9999",
            }
        ]
    )

    assert report["per_evidence"] == [
        {
            "evidence_id": "accepted",
            "peer_review_status": "peer_reviewed",
            "reasons": ["metadata_publication_type_peer_reviewed"],
        }
    ]


def test_supports_objects_tuple_payloads_and_nested_metadata():
    report = classify_evidence_peer_review_status(
        [
            EvidenceStub(
                id="object",
                title="Safety benchmark",
                metadata={"peer_review_status": "preprint", "venue": "Journal of Tests"},
            ),
            ({"id": "tuple", "metadata": {"review_status": "peer reviewed"}}, 0.8),
        ]
    )

    assert report["per_evidence"] == [
        {
            "evidence_id": "object",
            "peer_review_status": "preprint",
            "reasons": ["metadata_peer_review_status_preprint"],
        },
        {
            "evidence_id": "tuple",
            "peer_review_status": "peer_reviewed",
            "reasons": ["metadata_review_status_peer_reviewed"],
        },
    ]


def test_empty_input_is_deterministic():
    assert classify_evidence_peer_review_status([]) == {
        "total_evidence": 0,
        "status_counts": {
            "peer_reviewed": 0,
            "preprint": 0,
            "report": 0,
            "news_or_blog": 0,
            "documentation": 0,
            "unknown": 0,
        },
        "status_share": {
            "peer_reviewed": 0.0,
            "preprint": 0.0,
            "report": 0.0,
            "news_or_blog": 0.0,
            "documentation": 0.0,
            "unknown": 0.0,
        },
        "per_evidence": [],
    }
