from __future__ import annotations

from types import SimpleNamespace

from graph.store.relation_metadata_completeness_summary import summarize_relation_metadata_completeness


def test_relation_metadata_completeness_uses_defaults_and_top_level_fields():
    summary = summarize_relation_metadata_completeness(
        [
            {"type": "cites", "source": "notes", "confidence": 0.9, "metadata": {"evidence": "quote", "created_at": "2024-01-01"}},
            {"type": "cites", "source": "notes", "metadata": {"confidence": 0.8}},
            SimpleNamespace(relation_type="mentions", metadata={"source": "crm", "confidence": 1, "evidence": "x", "created_at": "2024-01-02"}),
        ]
    )

    rows = {(row["relation_type"], row["source"]): row for row in summary["rows"]}
    assert rows[("cites", "notes")]["complete_count"] == 1
    assert rows[("cites", "notes")]["incomplete_count"] == 1
    assert rows[("cites", "notes")]["missing_key_counts"] == [{"key": "created_at", "count": 1}, {"key": "evidence", "count": 1}]
    assert rows[("mentions", "crm")]["complete_count"] == 1


def test_relation_metadata_completeness_honors_custom_required_keys():
    summary = summarize_relation_metadata_completeness(
        [{"type": "links", "source": "web", "metadata": {"label": "A"}}, {"type": "links", "source": "web"}],
        required_keys=("label", "reviewed_by"),
    )

    row = summary["rows"][0]
    assert row["missing_metadata_count"] == 1
    assert row["missing_key_counts"] == [{"key": "label", "count": 1}, {"key": "reviewed_by", "count": 2}]
