from __future__ import annotations

from types import SimpleNamespace

from graph.rag.answer_entity_consistency import audit_answer_entity_consistency


def test_answer_entity_consistency_reports_supported_and_unsupported_entities():
    report = audit_answer_entity_consistency(
        "Ada Lovelace worked with Charles Babbage at Example Labs.",
        [
            {"id": "r1", "content": "Ada Lovelace wrote notes.", "metadata": {"organizations": ["Example Labs"]}},
            (SimpleNamespace(id="r2", metadata={"people": ["Charles Babbage"]}), 0.8),
        ],
    )

    assert report["answer_entities"] == ["Ada Lovelace", "Charles Babbage", "Example Labs"]
    assert report["evidence_entities"] == ["Ada Lovelace", "Charles Babbage", "Example Labs"]
    assert report["unsupported_entities"] == []
    assert report["support_ratio"] == 1.0
    assert report["supported_entities"][0]["entity"] == "Ada Lovelace"
    assert report["supported_entities"][0]["supporting_result_ids"] == ["r1"]
    assert {row["result_id"]: row["supported_entity_count"] for row in report["per_result_support"]} == {"r1": 2, "r2": 1}


def test_answer_entity_consistency_warns_for_empty_inputs_and_absent_entity():
    report = audit_answer_entity_consistency("Marie Curie cited Unknown Institute.", [])

    assert report["supported_entities"] == []
    assert report["unsupported_entities"] == [
        {"entity": "Marie Curie", "reason": "entity_absent_from_evidence"},
        {"entity": "Unknown Institute", "reason": "entity_absent_from_evidence"},
    ]
    assert report["warnings"] == ["no_evidence"]
    assert report["support_ratio"] == 0.0
    assert audit_answer_entity_consistency("", [])["warnings"] == ["empty_answer", "no_evidence"]
