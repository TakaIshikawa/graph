from __future__ import annotations

from types import SimpleNamespace

from graph.rag.answer_verification_plan import build_answer_verification_plan


def test_answer_verification_plan_prioritizes_missing_citations_and_dates():
    plan = build_answer_verification_plan(
        "revenue growth 2024",
        [
            {
                "id": "r1",
                "content": "Revenue grew 42 percent in 2024.",
                "source": "internal",
                "metadata": {"published_at": "2024-04-01"},
            },
            {
                "id": "r2",
                "content": "Revenue growth was discussed by analysts.",
                "url": "https://example.com/report",
                "source": "external",
            },
        ],
    )

    assert plan["query"] == "revenue growth 2024"
    assert plan["counts"] == {
        "result_count": 2,
        "check_count": 3,
        "with_citations": 1,
        "with_dates": 1,
        "with_provenance": 2,
        "source_count": 2,
    }
    assert plan["checks"][0]["id"] == "check-01-citations"
    assert plan["checks"][0]["target_result_ids"] == ["r1"]
    assert {key for key in plan["checks"][0]} == {
        "id",
        "priority",
        "reason",
        "target_result_ids",
        "suggested_action",
    }
    assert [check["id"] for check in plan["checks"]][:3] == [
        "check-01-citations",
        "check-02-facts",
        "check-03-dates",
    ]


def test_answer_verification_plan_accepts_objects_and_limits_checks():
    result = SimpleNamespace(
        id="obj-1",
        text="The launch moved to 2025 with 12 pilot customers.",
        metadata={"source_project": "roadmap"},
    )

    plan = build_answer_verification_plan("launch customers", [result], max_checks=2)

    assert plan["counts"]["result_count"] == 1
    assert len(plan["checks"]) == 2
    assert plan["checks"][0]["id"] == "check-01-citations"
    assert plan["checks"][1]["id"] == "check-02-dates"


def test_answer_verification_plan_empty_and_malformed_inputs_do_not_raise():
    assert build_answer_verification_plan("anything", [])["checks"] == []
    assert build_answer_verification_plan(None, None)["counts"]["result_count"] == 0
    assert build_answer_verification_plan("anything", object())["summary"].startswith(
        "No verification checks generated"
    )
