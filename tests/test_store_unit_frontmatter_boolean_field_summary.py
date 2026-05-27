from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_unit_frontmatter_boolean_fields


def test_summarize_unit_frontmatter_boolean_fields_distinguishes_values():
    summary = summarize_unit_frontmatter_boolean_fields(
        [
            {"id": "a", "metadata": {"draft": True, "published": "yes", "ready": "enabled"}},
            SimpleNamespace(id="b", metadata={"draft": False, "published": "no", "title": "Nope"}),
            {"id": "c", "content": "---\nflag: on\nblank: \n---\nBody"},
        ]
    )

    assert [(row["key"], row["true_count"], row["false_count"], row["string_boolean_count"], row["invalid_boolean_like_count"]) for row in summary["boolean_fields"]] == [
        ("draft", 1, 1, 0, 0),
        ("flag", 0, 0, 1, 0),
        ("published", 0, 0, 2, 0),
        ("ready", 0, 0, 0, 1),
    ]
