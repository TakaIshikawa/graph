from __future__ import annotations

from graph.rag.context_license_filter_plan import plan_context_license_filter


def test_context_license_filter_plan_recognizes_common_licenses():
    plan = plan_context_license_filter(
        [
            {"id": "pd", "metadata": {"license": "Public domain"}},
            {"id": "by", "metadata": {"license": "CC-BY 4.0"}},
            {"id": "nc", "metadata": {"license": "CC-BY-NC"}},
            {"id": "arr", "metadata": {"rights": "All rights reserved"}},
        ]
    )

    by_id = {item["item_id"]: item for item in plan["items"]}
    assert by_id["pd"]["allowed_uses"] == ["answer_generation", "quotation", "redistribution"]
    assert by_id["by"]["cautions"] == ["attribution_required"]
    assert by_id["nc"]["exclude_from_redistribution"] is True
    assert by_id["arr"]["allowed_uses"] == ["answer_generation"]
    assert set(plan["excluded_item_ids"]) == {"nc", "arr"}


def test_context_license_filter_plan_defaults_unknown_conservatively():
    plan = plan_context_license_filter([{"id": "unknown", "text": "No rights statement."}])

    assert plan["items"][0]["license"] == "unknown"
    assert plan["items"][0]["allowed_uses"] == ["answer_generation"]
    assert plan["items"][0]["exclude_from_redistribution"] is True
    assert "unknown_license" in plan["warnings"]
