from __future__ import annotations

from graph.rag.answer_recommendation_support_audit import audit_answer_recommendation_support


class EvidenceRecord:
    def __init__(self, text: str) -> None:
        self.text = text


def test_answer_recommendation_support_counts_supported_and_unsupported():
    audit = audit_answer_recommendation_support(
        "Teams should use staged rollout for database migrations. Avoid Friday releases.",
        [{"content": "A staged rollout reduces database migration incident impact."}],
    )

    assert audit["recommendation_count"] == 2
    assert audit["supported_count"] == 1
    assert audit["unsupported_recommendations"] == ["Avoid Friday releases."]
    assert audit["severity"] == "medium"


def test_answer_recommendation_support_accepts_strings_and_objects():
    audit = audit_answer_recommendation_support(
        "We recommend offline backups. The best option is tested restore drills.",
        [
            "Offline backups protect recovery paths.",
            EvidenceRecord("Tested restore drills are the strongest recovery option."),
        ],
    )

    assert audit["recommendation_count"] == 2
    assert audit["supported_count"] == 2
    assert audit["severity"] == "none"


def test_answer_recommendation_support_deduplicates_sentences():
    audit = audit_answer_recommendation_support(
        "You should rotate keys monthly. You should rotate keys monthly.",
        [],
    )

    assert audit["recommendation_count"] == 1
    assert audit["unsupported_recommendations"] == ["You should rotate keys monthly."]
    assert audit["severity"] == "high"
