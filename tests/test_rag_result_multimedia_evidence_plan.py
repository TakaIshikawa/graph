from __future__ import annotations

import pytest

from graph.rag.result_multimedia_evidence_plan import plan_result_multimedia_evidence


def test_multimedia_evidence_plan_detects_query_and_result_coverage():
    plan = plan_result_multimedia_evidence(
        "Show a chart, map, and table of store locations",
        [{"id": "chart", "metadata": {"media_types": ["chart"]}}],
    )

    assert plan["required_media_types"] == ["chart", "map", "table"]
    assert plan["present_media_types"] == ["chart"]
    assert plan["missing_media_types"] == ["map", "table"]
    assert plan["warnings"] == ["missing_required_media"]


def test_multimedia_evidence_plan_uses_expected_format_and_metadata():
    plan = plan_result_multimedia_evidence(
        "Summarize the hearing",
        [{"id": "video", "metadata": {"attachments": [{"type": "video"}]}}],
        expected_format="include video and audio",
    )

    assert plan["required_media_types"] == ["audio", "video"]
    assert plan["present_media_types"] == ["video"]
    assert plan["missing_media_types"] == ["audio"]


@pytest.mark.parametrize("query", ["", "  ", None])
def test_multimedia_evidence_plan_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        plan_result_multimedia_evidence(query)  # type: ignore[arg-type]
