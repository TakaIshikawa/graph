from __future__ import annotations

from graph.store.collection_member_count_summary import summarize_collection_member_counts


def test_collection_member_count_summary_buckets_by_source_and_type():
    summary = summarize_collection_member_counts(
        [
            {"source": "docs", "type": "folder"},
            {"source": "docs", "type": "folder", "member_ids": ["u1"]},
            {"source": "docs", "type": "folder", "member_ids": ["u1", "u2", "u3"]},
            {"source": "docs", "type": "folder", "member_ids": list(range(6))},
            {"source": "docs", "type": "folder", "member_ids": list(range(21))},
            {"source": "crm", "type": "list", "metadata": {"unit_ids": ["u1", "u2"]}},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "crm",
            "collection_type": "list",
            "collection_count": 1,
            "empty_count": 0,
            "singleton_count": 0,
            "small_count": 1,
            "medium_count": 0,
            "large_count": 0,
            "min_members": 2,
            "max_members": 2,
            "average_members": 2.0,
        },
        {
            "source": "docs",
            "collection_type": "folder",
            "collection_count": 5,
            "empty_count": 1,
            "singleton_count": 1,
            "small_count": 1,
            "medium_count": 1,
            "large_count": 1,
            "min_members": 0,
            "max_members": 21,
            "average_members": 6.2,
        },
    ]


def test_collection_member_count_summary_empty_input():
    assert summarize_collection_member_counts([]) == {"rows": [], "row_count": 0, "collection_count": 0}
