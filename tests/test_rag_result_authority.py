from __future__ import annotations

from dataclasses import dataclass

from graph.rag.result_authority import score_result_authority


@dataclass
class ResultStub:
    id: str
    metadata: dict


def by_id(rows: list[dict], result_id: str) -> dict:
    return next(row for row in rows if row["result_id"] == result_id)


def test_score_result_authority_rewards_stronger_authority_signals():
    rows = score_result_authority(
        [
            {
                "id": "paper",
                "source_type": "peer_reviewed",
                "author": "Ada Lovelace",
                "venue": "Nature",
                "url": "https://nature.com/articles/example",
                "citation_count": 250,
                "confidence": 0.95,
            },
            {
                "id": "post",
                "source_type": "blog",
                "url": "https://example.com/post",
                "confidence": 0.4,
            },
        ]
    )

    assert by_id(rows, "paper")["authority_score"] > by_id(rows, "post")["authority_score"]
    assert by_id(rows, "paper") == {
        "result_id": "paper",
        "authority_score": 1.0,
        "signals": [
            "baseline authority (+0.25)",
            "source type peer_reviewed (+0.22)",
            "author present (+0.12)",
            "venue Nature (+0.10)",
            "authority domain nature.com (+0.12)",
            "citations 250 (+0.16)",
            "confidence 0.95 (+0.12)",
        ],
        "warnings": [],
    }


def test_score_result_authority_warns_about_missing_metadata():
    rows = score_result_authority([{"id": "thin"}])

    assert rows == [
        {
            "result_id": "thin",
            "authority_score": 0.25,
            "signals": ["baseline authority (+0.25)"],
            "warnings": [
                "missing source type",
                "missing author",
                "missing publication venue",
                "missing URL or domain",
                "missing citation count",
                "missing confidence",
            ],
        }
    ]


def test_score_result_authority_supports_objects_nested_metadata_and_stable_ordering():
    rows = score_result_authority(
        [
            ResultStub(
                id="object",
                metadata={
                    "source_type": "government",
                    "authors": ["A", "B"],
                    "publisher": "NASA",
                    "domain": "nasa.gov",
                    "citations": ["x", "y"],
                    "confidence": "90",
                },
            ),
            {
                "unit": {
                    "id": "nested",
                    "metadata": {
                        "source_type": "news",
                        "byline": "Reporter",
                        "publication": "Daily",
                        "url": "https://daily.example/story",
                        "citation_count": 0,
                        "score": 0.5,
                    },
                }
            },
        ]
    )

    assert [row["result_id"] for row in rows] == ["object", "nested"]
    assert by_id(rows, "object")["authority_score"] == 1.0
    assert by_id(rows, "nested")["authority_score"] == 0.655
    assert by_id(rows, "nested")["warnings"] == ["zero citations"]
