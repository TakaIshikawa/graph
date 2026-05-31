from graph.store import summarize_unit_frontmatter_alias_collisions


def test_frontmatter_alias_collision_summary_reports_case_insensitive_collisions():
    units = [
        {"id": "a", "title": "A", "metadata": {"aliases": ["Shared", "Solo"]}},
        {"id": "b", "title": "B", "metadata": {"alias": "shared"}},
    ]

    result = summarize_unit_frontmatter_alias_collisions(units)

    assert result["collision_count"] == 1
    assert result["collisions"][0]["normalized_alias"] == "shared"
    assert result["collisions"][0]["unit_ids"] == ["a", "b"]
