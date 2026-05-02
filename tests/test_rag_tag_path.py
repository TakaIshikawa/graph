from __future__ import annotations

import pytest

from graph.rag import plan_tag_reading_path
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str | None = None,
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title or unit_id,
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata=metadata or {},
    )


def unit_ids(result: dict) -> list[str]:
    return [item["unit_id"] for item in result["units"]]


def test_plan_tag_reading_path_groups_units_by_tag_continuity():
    units = [
        unit("market", "Market", tags=["market"]),
        unit("battery", "Battery", tags=["storage", "solar"]),
        unit("grid", "Grid", tags=["storage", "grid"]),
        unit("solar", "Solar", tags=["solar"]),
    ]

    result = plan_tag_reading_path(reversed(units), start_tags=["solar"])

    assert unit_ids(result) == ["battery", "solar", "grid", "market"]
    assert result["units"][0]["matched_tags"] == ["solar"]
    assert result["units"][0]["transition_reason"] == "start_tag_match"
    assert result["units"][1]["matched_tags"] == ["solar"]
    assert result["units"][1]["transition_reason"] == "tag_continuity"
    assert result["units"][2]["matched_tags"] == ["storage"]
    assert result["units"][2]["previous_unit_id"] == "solar"


def test_plan_tag_reading_path_falls_back_deterministically_without_matches():
    units = [
        unit("unit-c", "Same", tags=["gamma"]),
        unit("unit-a", "Alpha", tags=["alpha"]),
        unit("unit-b", "Same", tags=["beta"]),
    ]

    first = plan_tag_reading_path(units, start_tags=["missing"])
    second = plan_tag_reading_path(reversed(units), start_tags=["missing"])

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-b", "unit-c"]
    assert first["units"][0]["transition_reason"] == "fallback"


def test_plan_tag_reading_path_prefers_unread_on_fallback_when_requested():
    units = [
        unit("read", "Alpha", metadata={"read_status": "read"}),
        unit("unread", "Beta", metadata={"read_status": "unread"}),
    ]

    result = plan_tag_reading_path(units, start_tags=["missing"], prefer_unread=True)

    assert unit_ids(result) == ["unread", "read"]
    assert result["units"][0]["transition_reason"] == "unread_fallback"
    assert result["stats"]["prefer_unread"] is True


def test_plan_tag_reading_path_honors_max_units_and_reports_stats():
    result = plan_tag_reading_path(
        [
            unit("unit-a", tags=["solar"]),
            unit("unit-b", tags=["solar"]),
            unit("unit-c", tags=["storage"]),
        ],
        start_tags=["solar"],
        max_units=2,
    )

    assert unit_ids(result) == ["unit-a", "unit-b"]
    assert result["stats"]["planned_units"] == 2
    assert result["stats"]["candidate_units"] == 3
    assert result["stats"]["omitted_units"] == 1
    assert result["stats"]["max_units"] == 2


def test_plan_tag_reading_path_returns_deterministic_empty_result():
    result = plan_tag_reading_path([], start_tags=["Solar"], max_units=0)

    assert result == {
        "units": [],
        "stats": {
            "total_units": 0,
            "candidate_units": 0,
            "planned_units": 0,
            "omitted_units": 0,
            "start_tags": ["Solar"],
            "start_tag_keys": ["solar"],
            "max_units": 0,
            "prefer_unread": False,
        },
    }


def test_plan_tag_reading_path_handles_untagged_inputs_deterministically():
    first = plan_tag_reading_path(
        [
            {"id": "unit-b", "title": "Beta"},
            {"id": "unit-a", "title": "Alpha", "tags": []},
        ]
    )
    second = plan_tag_reading_path(
        [
            {"id": "unit-a", "title": "Alpha", "tags": []},
            {"id": "unit-b", "title": "Beta"},
        ]
    )

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-b"]
    assert [item["matched_tags"] for item in first["units"]] == [[], []]
    assert [item["transition_reason"] for item in first["units"]] == [
        "initial_fallback",
        "initial_fallback",
    ]


def test_plan_tag_reading_path_accepts_mixed_tag_metadata_shapes_and_does_not_mutate():
    dict_unit = {
        "unit": {
            "id": "nested",
            "title": "Nested",
            "tags": [{"name": "Solar"}, "storage"],
            "metadata": {"tags": "ignored"},
        }
    }
    metadata_unit = {
        "id": "metadata",
        "title": "Metadata",
        "metadata": {"tags": ["storage", {"label": "Grid"}]},
    }
    original_metadata = list(metadata_unit["metadata"]["tags"])

    result = plan_tag_reading_path([metadata_unit, dict_unit], start_tags=("solar",))

    assert unit_ids(result) == ["nested", "metadata"]
    assert result["units"][0]["tags"] == ["Solar", "storage"]
    assert result["units"][1]["matched_tags"] == ["storage"]
    assert metadata_unit["metadata"]["tags"] == original_metadata


def test_plan_tag_reading_path_is_importable_from_graph_rag():
    from graph.rag import plan_tag_reading_path as imported

    assert imported is plan_tag_reading_path


@pytest.mark.parametrize("max_units", [-1, "2", True])
def test_plan_tag_reading_path_validates_max_units(max_units):
    with pytest.raises(ValueError, match="max_units must be a non-negative integer"):
        plan_tag_reading_path([], max_units=max_units)
