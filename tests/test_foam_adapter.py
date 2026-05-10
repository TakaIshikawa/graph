from __future__ import annotations

from pathlib import Path

from graph.adapters.foam import FoamWorkspaceAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_parses_markdown_note(tmp_path):
    _write(tmp_path / "hello.md", "# Hello World\n\nSome content here.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FOAM_WORKSPACE
    assert unit.source_entity_type == "note"
    assert unit.title == "Hello World"
    assert "Some content here." in unit.content
    assert unit.content_type == ContentType.ARTIFACT


def test_uses_filename_when_no_heading(tmp_path):
    _write(tmp_path / "my-note.md", "Just body text, no heading.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert result.units[0].title == "my-note"


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------

def test_parses_yaml_frontmatter(tmp_path):
    _write(
        tmp_path / "note.md",
        "---\ntitle: My Title\ntags:\n  - foo\n  - bar\ncustom: value\n---\n\nBody text.",
    )

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert unit.title == "My Title"
    assert "foo" in unit.tags
    assert "bar" in unit.tags
    assert unit.metadata["frontmatter"]["custom"] == "value"


def test_frontmatter_title_overrides_heading(tmp_path):
    _write(
        tmp_path / "note.md",
        "---\ntitle: FM Title\n---\n\n# Heading Title\n\nBody.",
    )

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()
    assert result.units[0].title == "FM Title"


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------

def test_extracts_wikilinks_as_relations(tmp_path):
    _write(tmp_path / "a.md", "# Note A\n\nSee [[b]] for details.")
    _write(tmp_path / "b.md", "# Note B\n\nContent of B.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.metadata["relation_type"] == "wikilink"


def test_wikilink_with_alias(tmp_path):
    _write(tmp_path / "source.md", "# Source\n\nLink to [[target|alias text]].")
    _write(tmp_path / "target.md", "# Target\n\nTarget content.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.edges) == 1
    assert result.edges[0].relation == EdgeRelation.REFERENCES


def test_wikilink_with_path(tmp_path):
    _write(tmp_path / "index.md", "# Index\n\nSee [[subdir/deep]].")
    _write(tmp_path / "subdir" / "deep.md", "# Deep Note\n\nNested content.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.edges) == 1


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_extracts_inline_tags(tmp_path):
    _write(tmp_path / "tagged.md", "# Tagged\n\nSome #topic and #another-tag here.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    tags = result.units[0].tags
    assert "topic" in tags
    assert "another-tag" in tags


def test_combines_frontmatter_and_inline_tags(tmp_path):
    _write(
        tmp_path / "mixed.md",
        "---\ntags:\n  - from-fm\n---\n\nInline #from-body text.",
    )

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    tags = result.units[0].tags
    assert "from-fm" in tags
    assert "from-body" in tags


# ---------------------------------------------------------------------------
# Note embeds
# ---------------------------------------------------------------------------

def test_extracts_embeds(tmp_path):
    _write(tmp_path / "parent.md", "# Parent\n\n![[child]]")
    _write(tmp_path / "child.md", "# Child\n\nEmbedded content.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.relation == EdgeRelation.CONTAINS
    assert edge.metadata["relation_type"] == "embed"
    # Embeds also stored in metadata
    parent = [u for u in result.units if "Parent" in u.title][0]
    assert "child" in parent.metadata["embeds"]


# ---------------------------------------------------------------------------
# Link reference definitions
# ---------------------------------------------------------------------------

def test_extracts_link_definitions(tmp_path):
    _write(
        tmp_path / "note.md",
        "# Note\n\nSee [reference].\n\n[reference]: path/to/target.md\n",
    )

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert "link_definitions" in unit.metadata
    assert unit.metadata["link_definitions"]["reference"] == "path/to/target.md"


# ---------------------------------------------------------------------------
# Daily notes
# ---------------------------------------------------------------------------

def test_daily_notes_classification(tmp_path):
    daily_dir = tmp_path / ".foam" / "daily"
    _write(daily_dir / "2024-01-15.md", "# Daily Note\n\nToday's thoughts.")
    _write(tmp_path / "regular.md", "# Regular Note\n\nNormal content.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    types = {u.source_entity_type for u in result.units}
    assert "daily_note" in types
    assert "note" in types

    daily = [u for u in result.units if u.source_entity_type == "daily_note"][0]
    assert daily.metadata["date"] == "2024-01-15"


def test_filter_by_entity_type(tmp_path):
    _write(tmp_path / ".foam" / "daily" / "2024-01-15.md", "Daily content.")
    _write(tmp_path / "regular.md", "# Regular\n\nContent.")

    notes_only = FoamWorkspaceAdapter(path=str(tmp_path)).ingest(entity_types=["note"])
    assert all(u.source_entity_type == "note" for u in notes_only.units)

    daily_only = FoamWorkspaceAdapter(path=str(tmp_path)).ingest(entity_types=["daily_note"])
    assert all(u.source_entity_type == "daily_note" for u in daily_only.units)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_directory(tmp_path):
    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_nonexistent_path(tmp_path):
    result = FoamWorkspaceAdapter(path=str(tmp_path / "nonexistent")).ingest()
    assert len(result.units) == 0


def test_no_path():
    result = FoamWorkspaceAdapter(path="").ingest()
    assert len(result.units) == 0


def test_skips_non_markdown_files(tmp_path):
    _write(tmp_path / "image.png", "not really an image")
    _write(tmp_path / "note.md", "# Note\n\nContent.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()
    assert len(result.units) == 1


def test_unresolved_wikilink_no_edge(tmp_path):
    _write(tmp_path / "orphan.md", "# Orphan\n\nLink to [[nonexistent]].")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 0


def test_recursive_directory_scan(tmp_path):
    _write(tmp_path / "root.md", "# Root")
    _write(tmp_path / "sub" / "nested.md", "# Nested")
    _write(tmp_path / "sub" / "deep" / "leaf.md", "# Leaf")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()
    assert len(result.units) == 3


def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("foam_workspace", path="/tmp/fake")
    assert adapter.name == "foam_workspace"


def test_metadata_includes_relative_path(tmp_path):
    _write(tmp_path / "docs" / "guide.md", "# Guide\n\nContent.")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert unit.metadata["relative_path"] == "docs/guide.md"


def test_multiple_wikilinks_in_one_note(tmp_path):
    _write(tmp_path / "hub.md", "# Hub\n\n[[a]] and [[b]] and [[a]] again.")
    _write(tmp_path / "a.md", "# A")
    _write(tmp_path / "b.md", "# B")

    result = FoamWorkspaceAdapter(path=str(tmp_path)).ingest()

    # Should have edges for both a and b (duplicates resolved by regex findall)
    ref_edges = [e for e in result.edges if e.relation == EdgeRelation.REFERENCES]
    target_ids = {e.to_unit_id for e in ref_edges}
    assert len(target_ids) == 2
