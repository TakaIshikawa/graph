from __future__ import annotations

from types import SimpleNamespace

from graph.store.collection_metadata_drift_summary import summarize_collection_metadata_drift


def test_collection_metadata_drift_includes_empty_metadata_schema_variant():
    summary = summarize_collection_metadata_drift([{"source": "notes", "type": "folder"}])

    assert summary == {
        "total_collections": 1,
        "rows": [
            {
                "source": "notes",
                "collection_type": "folder",
                "collection_count": 1,
                "distinct_metadata_key_count": 0,
                "common_keys": [],
                "rare_keys": [],
                "schema_variants": [{"keys": [], "count": 1}],
            }
        ],
    }


def test_collection_metadata_drift_counts_repeated_schemas_and_key_frequency():
    summary = summarize_collection_metadata_drift(
        [
            {"source": "notes", "type": "folder", "metadata": {"owner": "a", "status": "open"}},
            {"source": "notes", "type": "folder", "metadata": {"owner": "b", "status": "closed"}},
            {"source": "notes", "type": "folder", "metadata": {"owner": "c", "priority": "high"}},
        ]
    )

    row = summary["rows"][0]
    assert row["collection_count"] == 3
    assert row["distinct_metadata_key_count"] == 3
    assert row["common_keys"] == [{"key": "owner", "count": 3}]
    assert row["rare_keys"] == [{"key": "priority", "count": 1}]
    assert row["schema_variants"] == [
        {"keys": ["owner", "priority"], "count": 1},
        {"keys": ["owner", "status"], "count": 2},
    ]


def test_collection_metadata_drift_groups_objects_by_source_and_type_deterministically():
    summary = summarize_collection_metadata_drift(
        [
            SimpleNamespace(source="zeta", collection_type="topic", metadata={"B": 1}),
            {"metadata": {"source": "alpha", "collection_type": "topic", "A": 1}},
            {"source": "alpha", "type": "archive", "metadata": {"A": 1}},
        ]
    )

    assert [(row["source"], row["collection_type"]) for row in summary["rows"]] == [
        ("alpha", "archive"),
        ("alpha", "topic"),
        ("zeta", "topic"),
    ]
