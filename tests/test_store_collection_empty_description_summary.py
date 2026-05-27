from __future__ import annotations

from types import SimpleNamespace

from graph.store.collection_empty_description_summary import summarize_collection_empty_descriptions


def test_collection_empty_description_summary_groups_empty_descriptions():
    summary = summarize_collection_empty_descriptions(
        [
            {"id": "c2", "description": " ", "source": "manual"},
            {"id": "c1", "description": None, "type": "project"},
            {"id": "c3", "description": "Done", "source": "manual"},
        ]
    )

    assert summary == {
        "total_collections": 3,
        "empty_description_count": 2,
        "described_count": 1,
        "affected_collection_ids": ["c1", "c2"],
        "counts_by_source": [{"source": "manual", "count": 1}, {"source": "project", "count": 1}],
    }


def test_collection_empty_description_summary_supports_objects_and_metadata():
    summary = summarize_collection_empty_descriptions([SimpleNamespace(collection_id="c", metadata={"source": "import"})])

    assert summary["counts_by_source"] == [{"source": "import", "count": 1}]
