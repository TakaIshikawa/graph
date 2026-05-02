from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_units_to_org
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=content_type,
        tags=tags or [],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_to_org_writes_outline_metadata_and_stats(tmp_path):
    path = tmp_path / "nested" / "units.org"

    stats = export_units_to_org(
        [
            unit(
                "unit-a",
                "Alpha",
                "Plain body",
                ["storage", "solar"],
                source_project=SourceProject.PINBOARD,
                content_type=ContentType.FINDING,
            )
        ],
        path,
        title="Research KB",
    )
    text = path.read_text(encoding="utf-8")

    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "units_exported": 1,
        "bytes_written": len(text.encode("utf-8")),
    }
    assert text == (
        "#+TITLE: Research KB\n"
        "\n"
        "* Alpha\n"
        ":PROPERTIES:\n"
        ":source_project: pinboard\n"
        ":source_id: source-unit-a\n"
        ":content_type: finding\n"
        ":tags: solar, storage\n"
        ":created_at: 2026-05-01T10:15:00+00:00\n"
        ":updated_at: 2026-05-01T10:15:00+00:00\n"
        ":END:\n"
        "\n"
        "Plain body\n"
    )


def test_export_units_to_org_preserves_materialized_sequence_order(tmp_path):
    path = tmp_path / "units.org"

    export_units_to_org(
        [
            unit("unit-c", "Gamma", "Gamma content"),
            unit("unit-a", "Alpha", "Alpha content"),
        ],
        path,
    )
    text = path.read_text(encoding="utf-8")

    assert text.index("* Gamma") < text.index("* Alpha")


def test_export_units_to_org_sorts_non_sequence_units_by_title_then_id(tmp_path):
    path = tmp_path / "units.org"
    units = (
        item
        for item in [
            unit("unit-c", "Gamma", "Gamma content"),
            unit("unit-b", "Alpha", "Alpha B content"),
            unit("unit-a", "Alpha", "Alpha A content"),
        ]
    )

    export_units_to_org(units, path)
    text = path.read_text(encoding="utf-8")

    assert text.index("Alpha A content") < text.index("Alpha B content")
    assert text.index("Alpha B content") < text.index("* Gamma")


def test_export_units_to_org_can_omit_metadata(tmp_path):
    path = tmp_path / "without-metadata.org"

    stats = export_units_to_org(
        [unit("unit-a", "Alpha", "Alpha content", ["tag-a"])],
        path,
        include_metadata=False,
    )
    text = path.read_text(encoding="utf-8")

    assert stats["units_exported"] == 1
    assert "* Alpha" in text
    assert "Alpha content" in text
    assert ":PROPERTIES:" not in text
    assert ":source_id:" not in text
    assert ":tags:" not in text


def test_export_units_to_org_max_units_keeps_complete_headings(tmp_path):
    path = tmp_path / "limited.org"

    stats = export_units_to_org(
        [
            unit("unit-a", "Alpha", "Alpha content"),
            unit("unit-b", "Beta", "Beta content"),
            unit("unit-c", "Gamma", "Gamma content"),
        ],
        path,
        max_units=2,
    )
    text = path.read_text(encoding="utf-8")

    assert stats["units_scanned"] == 3
    assert stats["units_exported"] == 2
    assert "* Alpha" in text
    assert "* Beta" in text
    assert "* Gamma" not in text
    assert "Gamma content" not in text


def test_export_units_to_org_allows_zero_max_units(tmp_path):
    path = tmp_path / "empty.org"

    stats = export_units_to_org(
        [unit("unit-a", "Alpha", "Alpha content")],
        path,
        max_units=0,
    )
    text = path.read_text(encoding="utf-8")

    assert stats["units_scanned"] == 1
    assert stats["units_exported"] == 0
    assert text == "#+TITLE: Knowledge Graph\n\n# No units exported.\n"


def test_export_units_to_org_rejects_negative_max_units(tmp_path):
    with pytest.raises(ValueError, match="max_units must be a non-negative integer"):
        export_units_to_org([], tmp_path / "bad.org", max_units=-1)


def test_export_units_to_org_fences_content_with_org_sensitive_markup(tmp_path):
    path = tmp_path / "sensitive.org"

    export_units_to_org(
        [
            unit(
                "unit-a",
                "Alpha",
                "* Nested heading\n[[file:Beta.org][Beta]]\n#+END_EXAMPLE",
            )
        ],
        path,
    )
    text = path.read_text(encoding="utf-8")

    assert (
        "#+BEGIN_EXAMPLE\n"
        "* Nested heading\n"
        "[[file:Beta.org][Beta]]\n"
        ",#+END_EXAMPLE\n"
        "#+END_EXAMPLE"
    ) in text
