from __future__ import annotations

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.text_outline import TextOutlineAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def _titles(result) -> list[str]:
    return [unit.title for unit in result.units]


def _edge_title_pairs(result) -> list[tuple[str, str]]:
    titles = {unit.source_id: unit.title for unit in result.units}
    return [(titles[edge.from_unit_id], titles[edge.to_unit_id]) for edge in result.edges]


def test_text_outline_ingests_nested_indentation_with_contains_edges(tmp_path):
    outline = tmp_path / "plan.txt"
    outline.write_text(
        "\n".join(
            [
                "Research",
                "  Literature review",
                "    Collect papers",
                "  Prototype",
            ]
        ),
        encoding="utf-8",
    )

    result = TextOutlineAdapter(path=str(outline)).ingest()

    assert _titles(result) == [
        "Research",
        "Literature review",
        "Collect papers",
        "Prototype",
    ]
    assert _edge_title_pairs(result) == [
        ("Research", "Literature review"),
        ("Literature review", "Collect papers"),
        ("Research", "Prototype"),
    ]
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)

    child = result.units[1]
    assert child.source_project == SourceProject.TEXT_OUTLINE
    assert child.source_entity_type == "text_outline_item"
    assert child.content_type == ContentType.INSIGHT
    assert child.content == "  Literature review"
    assert child.tags == ["outline"]
    assert child.metadata["level"] == 2
    assert child.metadata["parent_title"] == "Research"


def test_text_outline_strips_common_bullet_markers_from_titles(tmp_path):
    outline = tmp_path / "bullets.txt"
    outline.write_text(
        "\n".join(
            [
                "- Root",
                "  * Star child",
                "  + Plus child",
                "  1. Numbered child",
                "  a) Letter child",
            ]
        ),
        encoding="utf-8",
    )

    result = TextOutlineAdapter(path=str(outline)).ingest()

    assert _titles(result) == [
        "Root",
        "Star child",
        "Plus child",
        "Numbered child",
        "Letter child",
    ]
    assert _edge_title_pairs(result) == [
        ("Root", "Star child"),
        ("Root", "Plus child"),
        ("Root", "Numbered child"),
        ("Root", "Letter child"),
    ]


def test_text_outline_ids_are_stable_for_same_file_content(tmp_path):
    outline = tmp_path / "stable.txt"
    outline.write_text("- Root\n  - Child\n", encoding="utf-8")

    first = TextOutlineAdapter(path=str(outline)).ingest()
    second = TextOutlineAdapter(path=str(outline)).ingest()

    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert all(unit.source_id.startswith("text_outline:") for unit in first.units)
    assert all(edge.id.startswith("text-outline-contains-") for edge in first.edges)


def test_text_outline_ignores_empty_and_comment_lines_without_breaking_hierarchy(tmp_path):
    outline = tmp_path / "comments.txt"
    outline.write_text(
        "\n".join(
            [
                "# ignored",
                "",
                "Root",
                "  // ignored too",
                "  Child",
                "    # ignored nested",
                "    Grandchild",
            ]
        ),
        encoding="utf-8",
    )

    result = TextOutlineAdapter(path=str(outline)).ingest()

    assert _titles(result) == ["Root", "Child", "Grandchild"]
    assert _edge_title_pairs(result) == [("Root", "Child"), ("Child", "Grandchild")]


def test_text_outline_adapter_is_registered():
    assert "text_outline" in list_adapters()
    adapter = get_adapter("text_outline", path="/tmp/outline.txt")
    assert isinstance(adapter, TextOutlineAdapter)
    assert adapter.name == "text_outline"
