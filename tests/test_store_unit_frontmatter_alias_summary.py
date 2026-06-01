from __future__ import annotations

from graph.store.unit_frontmatter_alias_summary import summarize_unit_frontmatter_aliases


def test_reads_aliases_and_alias_scalar_and_lists():
    summary = summarize_unit_frontmatter_aliases(
        [
            {"id": "u1", "content": "---\nalias: One\n---\nBody"},
            {"id": "u2", "content": "---\naliases:\n  - Two\n  - one\n---\nBody"},
        ]
    )

    assert summary["total_units"] == 2
    assert summary["units_with_aliases"] == 2
    assert summary["alias_count"] == 3
    assert summary["duplicate_aliases"] == ["One"]
    assert summary["samples"][:2] == [
        {"unit_id": "u1", "key": "alias", "alias": "One"},
        {"unit_id": "u2", "key": "aliases", "alias": "Two"},
    ]


def test_malformed_frontmatter_does_not_crash():
    summary = summarize_unit_frontmatter_aliases([{"id": "bad", "content": "---\naliases: [unterminated\n---"}])

    assert summary["alias_count"] == 0
    assert summary["samples"] == []
