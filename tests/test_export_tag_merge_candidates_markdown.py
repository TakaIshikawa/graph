from __future__ import annotations

from graph.export.tag_merge_candidates_markdown import export_tag_merge_candidates_markdown
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project A",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata={},
        tags=tags,
    )


def test_tag_merge_candidates_markdown_empty_input_has_stable_message():
    assert export_tag_merge_candidates_markdown([]) == (
        "# Tag Merge Candidates\n\nNo tag merge candidates found.\n"
    )


def test_tag_merge_candidates_markdown_groups_case_punctuation_plural_and_space_variants():
    text = export_tag_merge_candidates_markdown(
        [
            unit("u3", ["Knowledge Graphs", "Solo"]),
            unit("u1", ["knowledge-graph", "AI"]),
            unit("u2", ["Knowledge graph", "a.i."]),
            unit("u4", ["A I"]),
        ]
    )

    assert text == (
        "# Tag Merge Candidates\n"
        "\n"
        "## a i\n"
        "\n"
        "- Suggested canonical tag: `AI`\n"
        "- Raw variants:\n"
        "- `A I` - 1 unit(s); examples: `u4`\n"
        "- `a.i.` - 1 unit(s); examples: `u2`\n"
        "- `AI` - 1 unit(s); examples: `u1`\n"
        "\n"
        "## knowledge graph\n"
        "\n"
        "- Suggested canonical tag: `Knowledge graph`\n"
        "- Raw variants:\n"
        "- `Knowledge graph` - 1 unit(s); examples: `u2`\n"
        "- `Knowledge Graphs` - 1 unit(s); examples: `u3`\n"
        "- `knowledge-graph` - 1 unit(s); examples: `u1`\n"
    )


def test_tag_merge_candidates_markdown_counts_units_examples_and_writes_path(tmp_path):
    units = [
        {"id": "u2", "tags": ["Project Plan", "project-plan"]},
        {"id": "u1", "tags": ["project plan"]},
        {"id": "u3", "tags": ["project-plan"]},
    ]
    path = tmp_path / "reports" / "tag-merge.md"

    expected = export_tag_merge_candidates_markdown(units)
    stats = export_tag_merge_candidates_markdown(units, path)

    assert "- Suggested canonical tag: `project-plan`" in expected
    assert "- `project-plan` - 2 unit(s); examples: `u2`, `u3`" in expected
    assert "- `Project Plan` - 1 unit(s); examples: `u2`" in expected
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 3,
        "candidate_group_count": 1,
        "bytes_written": path.stat().st_size,
    }
