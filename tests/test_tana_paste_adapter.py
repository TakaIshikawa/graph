from __future__ import annotations

import os
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.tana_paste import TanaPasteAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def _edge_title_pairs(result) -> list[tuple[str, str]]:
    titles = {unit.source_id: unit.title for unit in result.units}
    return [(titles[edge.from_unit_id], titles[edge.to_unit_id]) for edge in result.edges]


def test_tana_paste_ingests_nested_bullets_with_contains_edges(tmp_path):
    export = tmp_path / "notes.txt"
    export.write_text(
        "\n".join(
            [
                "- Project Alpha #strategy [[Roadmap]]",
                "\t- First milestone #planning",
                "\t\t- Draft success criteria [[Metric Catalog]]",
                "\t- Second milestone",
            ]
        ),
        encoding="utf-8",
    )

    result = TanaPasteAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == [
        "Project Alpha",
        "First milestone",
        "Draft success criteria",
        "Second milestone",
    ]
    assert _edge_title_pairs(result) == [
        ("Project Alpha", "First milestone"),
        ("First milestone", "Draft success criteria"),
        ("Project Alpha", "Second milestone"),
    ]
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)

    root = result.units[0]
    assert root.source_project == SourceProject.TANA_PASTE
    assert root.source_entity_type == "bullet"
    assert root.content_type == ContentType.ARTIFACT
    assert root.content == "Project Alpha #strategy [[Roadmap]]"
    assert root.tags == ["strategy"]
    assert root.metadata["tags"] == ["strategy"]
    assert root.metadata["references"] == ["Roadmap"]
    assert root.metadata["raw_title"] == "Project Alpha #strategy [[Roadmap]]"
    assert root.metadata["level"] == 1
    assert root.metadata["indent"] == 0

    grandchild = result.units[2]
    assert grandchild.metadata["parent_title"] == "First milestone"
    assert grandchild.metadata["references"] == ["Metric Catalog"]


def test_tana_paste_multiline_continuations_are_folded_into_previous_bullet(tmp_path):
    export = tmp_path / "continuation.txt"
    export.write_text(
        "\n".join(
            [
                "- Main idea #theme",
                "  Extra detail line references [[Long Note]]",
                "  Another continuation with #detail",
                "  - Child idea",
            ]
        ),
        encoding="utf-8",
    )

    result = TanaPasteAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Main idea", "Child idea"]
    assert _edge_title_pairs(result) == [("Main idea", "Child idea")]
    root = result.units[0]
    assert root.content == "\n".join(
        [
            "Main idea #theme",
            "Extra detail line references [[Long Note]]",
            "Another continuation with #detail",
        ]
    )
    assert root.tags == ["theme", "detail"]
    assert root.metadata["references"] == ["Long Note"]


def test_tana_paste_directory_ids_are_stable_and_sorted(tmp_path):
    first = tmp_path / "b.txt"
    second = tmp_path / "nested" / "a.txt"
    second.parent.mkdir()
    first.write_text("- B\n", encoding="utf-8")
    second.write_text("- A\n  - A child\n", encoding="utf-8")

    one = TanaPasteAdapter(path=str(tmp_path)).ingest()
    two = TanaPasteAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in one.units] == ["B", "A", "A child"]
    assert [unit.source_id for unit in one.units] == [unit.source_id for unit in two.units]
    assert [edge.id for edge in one.edges] == [edge.id for edge in two.edges]
    assert all(unit.source_id.startswith("tana_paste:") for unit in one.units)
    assert all(edge.id.startswith("tana-paste-contains-") for edge in one.edges)


def test_tana_paste_since_filter_uses_file_mtime(tmp_path):
    old_file = tmp_path / "old.txt"
    new_file = tmp_path / "new.txt"
    old_file.write_text("- Old\n", encoding="utf-8")
    new_file.write_text("- New\n", encoding="utf-8")
    old_mtime = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    new_mtime = datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp()
    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(new_file, (new_mtime, new_mtime))

    result = TanaPasteAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="tana_paste",
            source_entity_type="bullet",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["New"]
    assert result.edges == []


def test_tana_paste_entity_filter_and_missing_paths_return_empty_result(tmp_path):
    export = tmp_path / "notes.txt"
    export.write_text("- Ignored\n", encoding="utf-8")

    filtered = TanaPasteAdapter(path=str(export)).ingest(entity_types=["page"])
    missing = TanaPasteAdapter(path=str(tmp_path / "missing.txt")).ingest()

    assert filtered.units == []
    assert filtered.edges == []
    assert missing.units == []
    assert missing.edges == []


def test_tana_paste_adapter_is_registered():
    assert "tana_paste" in list_adapters()
    adapter = get_adapter("tana_paste", path="/tmp/tana.txt")
    assert isinstance(adapter, TanaPasteAdapter)
    assert adapter.name == "tana_paste"
