from __future__ import annotations

from dataclasses import dataclass

from graph.rag.claim_date_confidence import score_claim_date_confidence


@dataclass
class Claim:
    id: str
    claim: str
    metadata: dict


def test_claim_date_confidence_classifies_date_sources():
    report = score_claim_date_confidence(
        [
            {"id": "dated", "claim": "Revenue rose on 2025-03-01."},
            {"id": "inferred", "text": "Revenue rose.", "date": "2025-03-01"},
            {"id": "undated", "content": "Revenue rose."},
            {"id": "conflict", "claim": "Revenue rose in 2024.", "metadata": {"published_at": "2025-01-01"}},
        ]
    )

    assert [(row["claim_id"], row["label"], row["confidence"]) for row in report["claims"]] == [
        ("dated", "dated", 1.0),
        ("inferred", "inferred-date", 0.65),
        ("undated", "undated", 0.0),
        ("conflict", "conflicting-date", 0.25),
    ]
    assert report["summary"] == {"conflicting-date": 1, "dated": 1, "inferred-date": 1, "undated": 1}


def test_claim_date_confidence_accepts_objects_and_tuples():
    report = score_claim_date_confidence([(Claim("obj", "Filed in 2026.", {"date": "2026-05-01"}), 1.0)])

    assert report["claims"][0]["claim_years"] == ["2026"]
    assert report["claims"][0]["metadata_dates"] == ["2026-05-01"]
