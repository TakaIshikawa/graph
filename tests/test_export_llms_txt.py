from __future__ import annotations

import pytest

from graph.export import export_units_to_llms_txt
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def test_export_llms_txt_sorts_non_sequence_units_deterministically(tmp_path):
    units = (
        item
        for item in [
            unit("unit-c", "Gamma", "Gamma content"),
            unit("unit-b", "Alpha", "Alpha B content"),
            unit("unit-a", "Alpha", "Alpha A content", ["solar", "storage"]),
        ]
    )

    path = tmp_path / "llms.txt"
    stats = export_units_to_llms_txt(units, path, title="Research KB")
    text = path.read_text()

    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "units_exported": 3,
        "bytes_written": len(text.encode("utf-8")),
    }
    assert text.startswith("# Research KB\n\n## Index\n")
    assert text.index("[Alpha](#unit-a)") < text.index("[Alpha](#unit-b)")
    assert text.index("[Alpha](#unit-b)") < text.index("[Gamma](#unit-c)")
    assert '<a id="unit-a"></a>' in text
    assert "- Source: max/insight (`source-unit-a`)" in text
    assert "- Type: `insight`" in text
    assert "- Tags: `solar`, `storage`" in text
    assert "Alpha A content" in text


def test_export_llms_txt_preserves_materialized_sequence_order(tmp_path):
    units = [
        unit("unit-c", "Gamma", "Gamma content"),
        unit("unit-a", "Alpha", "Alpha content"),
    ]

    path = tmp_path / "ordered.txt"
    export_units_to_llms_txt(units, path)
    text = path.read_text()

    assert text.index("[Gamma](#unit-c)") < text.index("[Alpha](#unit-a)")


def test_export_llms_txt_can_omit_metadata(tmp_path):
    units = [unit("unit-a", "Alpha", "Alpha content", ["tag-a"])]

    path = tmp_path / "without-metadata.txt"
    stats = export_units_to_llms_txt(units, path, include_metadata=False)
    text = path.read_text()

    assert stats["units_exported"] == 1
    assert "### Alpha" in text
    assert "Alpha content" in text
    assert "- ID:" not in text
    assert "- Source:" not in text
    assert "- Type:" not in text
    assert "- Tags:" not in text


def test_export_llms_txt_max_units_keeps_complete_sections(tmp_path):
    units = [
        unit("unit-a", "Alpha", "Alpha content"),
        unit("unit-b", "Beta", "Beta content"),
        unit("unit-c", "Gamma", "Gamma content"),
    ]

    path = tmp_path / "limited.txt"
    stats = export_units_to_llms_txt(units, path, max_units=2)
    text = path.read_text()

    assert stats["units_scanned"] == 3
    assert stats["units_exported"] == 2
    assert "[Alpha](#unit-a)" in text
    assert "[Beta](#unit-b)" in text
    assert "[Gamma](#unit-c)" not in text
    assert "### Alpha" in text
    assert "### Beta" in text
    assert "### Gamma" not in text
    assert "Gamma content" not in text


def test_export_llms_txt_allows_zero_max_units(tmp_path):
    path = tmp_path / "empty.txt"
    stats = export_units_to_llms_txt(
        [unit("unit-a", "Alpha", "Alpha content")],
        path,
        max_units=0,
    )
    text = path.read_text()

    assert stats["units_scanned"] == 1
    assert stats["units_exported"] == 0
    assert "_No units exported._" in text
    assert "### Alpha" not in text


def test_export_llms_txt_rejects_negative_max_units(tmp_path):
    with pytest.raises(ValueError, match="max_units must be a non-negative integer"):
        export_units_to_llms_txt([], tmp_path / "bad.txt", max_units=-1)


def test_export_llms_txt_escapes_markdown_and_fences_content(tmp_path):
    units = [
        unit(
            "unit-a/b",
            "# API [draft](v2) \\ notes",
            "Content with ``` fenced text",
            ["tag`one"],
        )
    ]

    path = tmp_path / "escaped.txt"
    export_units_to_llms_txt(units, path, title="# Agent [KB]")
    text = path.read_text()

    assert r"# \# Agent [KB]" in text
    assert r"[# API \[draft\]\(v2\) \\ notes](#unit-a-b)" in text
    assert r"### \# API [draft](v2) \\ notes" in text
    assert r"- Tags: `tag\`one`" in text
    assert "````\nContent with ``` fenced text\n````" in text
