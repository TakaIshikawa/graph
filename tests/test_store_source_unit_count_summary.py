from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_source_unit_counts


def test_source_unit_counts_groups_sources_missing_and_ordering():
    summary = summarize_source_unit_counts(
        [
            {"id": "u1", "source_project": "readwise"},
            {"id": "u2", "metadata": {"source_project": "pocket"}},
            SimpleNamespace(id="u3", metadata={"source": {"name": "pocket"}}),
            {"id": "u4", "source": "readwise"},
            {"id": "u5", "source_project": ""},
            {"id": "u6"},
            {"id": "u7", "source_project": "apple"},
        ],
        top_limit=2,
    )

    assert summary == {
        "total_units": 7,
        "source_count": 3,
        "missing_source_units": 2,
        "source_counts": {"missing": 2, "pocket": 2, "readwise": 2, "apple": 1},
        "top_sources": [{"source": "missing", "unit_count": 2}, {"source": "pocket", "unit_count": 2}],
        "rows": [
            {"source": "missing", "unit_count": 2},
            {"source": "pocket", "unit_count": 2},
            {"source": "readwise", "unit_count": 2},
            {"source": "apple", "unit_count": 1},
        ],
    }
