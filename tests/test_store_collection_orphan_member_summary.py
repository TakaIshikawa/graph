from __future__ import annotations

from types import SimpleNamespace

from graph.store.collection_orphan_member_summary import summarize_collection_orphan_members


def test_collection_orphan_member_summary_reports_missing_members():
    summary = summarize_collection_orphan_members(
        [
            {"id": "c1", "title": "Roadmap", "member_ids": ["u1", "u3", "u3", {"unit_id": "u2"}]},
            {"id": "c2", "title": "Complete", "members": ["u1"]},
        ],
        [{"id": "u1"}, {"id": "u2"}],
    )

    assert summary == {
        "rows": [
            {
                "collection_id": "c1",
                "collection_title": "Roadmap",
                "missing_member_count": 1,
                "missing_member_ids": ["u3"],
                "resolved_member_count": 2,
            }
        ],
        "row_count": 1,
        "collection_count": 2,
    }


def test_collection_orphan_member_summary_supports_objects_metadata_and_empty_members():
    summary = summarize_collection_orphan_members(
        [
            SimpleNamespace(collection_id="c2", name="Nested", metadata={"items": [{"id": "u1"}, {"source_id": "u4"}]}),
            {"id": "c1"},
        ],
        [SimpleNamespace(id="u1")],
    )

    assert summary["rows"] == [
        {
            "collection_id": "c2",
            "collection_title": "Nested",
            "missing_member_count": 1,
            "missing_member_ids": ["u4"],
            "resolved_member_count": 1,
        }
    ]
