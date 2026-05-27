from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_collection_member_source_mix


def test_collection_member_source_mix_resolves_members_and_missing_ids():
    summary = summarize_collection_member_source_mix(
        [
            {"id": "c1", "member_ids": ["u1", "u2", "u4"]},
            SimpleNamespace(collection_id="c2", metadata={"items": [{"unit_id": "u3"}, {"id": "u9"}]}),
        ],
        [
            {"id": "u1", "source_project": "readwise"},
            SimpleNamespace(unit_id="u2", metadata={"source_project": "pocket"}),
            {"source_id": "u3", "metadata": {"source": "readwise"}},
        ],
    )

    assert summary == {
        "collection_count": 2,
        "rows": [
            {
                "collection_id": "c1",
                "total_members": 3,
                "matched_members": 2,
                "missing_members": 1,
                "dominant_source": "pocket",
                "source_counts": {"pocket": 1, "readwise": 1},
                "mixed_source": True,
            },
            {
                "collection_id": "c2",
                "total_members": 2,
                "matched_members": 1,
                "missing_members": 1,
                "dominant_source": "readwise",
                "source_counts": {"readwise": 1},
                "mixed_source": False,
            },
        ],
    }


def test_collection_member_source_mix_tie_breaks_dominant_source_by_name():
    summary = summarize_collection_member_source_mix(
        [{"id": "c1", "members": ["b", "a"]}],
        [{"id": "a", "source_project": "alpha"}, {"id": "b", "source_project": "beta"}],
    )

    assert summary["rows"][0]["dominant_source"] == "alpha"
