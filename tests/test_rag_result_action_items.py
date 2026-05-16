from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.result_action_items import extract_result_action_items


@dataclass
class Result:
    unit_id: str
    text: str = ""
    metadata: dict | None = None


def test_extract_result_action_items_recognizes_common_cues_in_order():
    rows = extract_result_action_items(
        [
            {
                "id": "a",
                "content": "\n".join(
                    [
                        "Context only.",
                        "TODO: confirm the citation source",
                        "- [ ] update the retrieval fixture",
                        "Next step: publish the evidence report.",
                    ]
                ),
            },
            {"id": "b", "text": "Follow up: ask owners for missing dates."},
        ]
    )

    assert rows == [
        {"result_id": "a", "action": "confirm the citation source", "cue": "todo", "position": 0},
        {"result_id": "a", "action": "update the retrieval fixture", "cue": "checkbox", "position": 1},
        {"result_id": "a", "action": "publish the evidence report.", "cue": "next step", "position": 2},
        {"result_id": "b", "action": "ask owners for missing dates.", "cue": "follow up", "position": 3},
    ]


def test_extract_result_action_items_supports_objects_tuples_and_metadata_fallbacks():
    rows = extract_result_action_items(
        [
            (Result("object", text="FIXME repair object parsing"), 0.8),
            {"metadata": {"source_id": "meta", "snippet": "Action item: check metadata snippets."}},
        ]
    )

    assert rows == [
        {"result_id": "object", "action": "repair object parsing", "cue": "fixme", "position": 0},
        {"result_id": "meta", "action": "check metadata snippets.", "cue": "action item", "position": 1},
    ]


def test_extract_result_action_items_normalizes_whitespace_and_strips_markdown_markers():
    rows = extract_result_action_items(
        [
            {
                "id": "a",
                "content": "- [x]   close   the loop\n* TODO   review   acceptance criteria",
            }
        ]
    )

    assert rows == [
        {"result_id": "a", "action": "close the loop", "cue": "checkbox", "position": 0},
        {"result_id": "a", "action": "review acceptance criteria", "cue": "todo", "position": 1},
    ]


def test_extract_result_action_items_respects_limit_and_empty_results():
    assert extract_result_action_items([{"content": "TODO one\nTODO two"}], max_items=1) == [
        {"result_id": "result-1", "action": "one", "cue": "todo", "position": 0}
    ]
    assert extract_result_action_items([{"content": "No task cue here."}]) == []
    assert extract_result_action_items([{"content": "TODO one"}], max_items=0) == []


@pytest.mark.parametrize("max_items", [-1, 1.5, False, "2"])
def test_extract_result_action_items_validates_max_items(max_items):
    with pytest.raises(ValueError, match="max_items"):
        extract_result_action_items([], max_items=max_items)
