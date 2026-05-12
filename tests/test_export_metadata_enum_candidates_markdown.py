from __future__ import annotations

import pytest

from graph.export import export_metadata_enum_candidates_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    metadata: dict,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_enum_candidates_flattens_nested_scalar_paths_and_reports_counts():
    text = export_metadata_enum_candidates_markdown(
        [
            unit("b", {"status": "closed", "review": {"state": "draft"}}, source_project="pinboard"),
            unit("a", {"status": "open", "review": {"state": "draft"}}),
            unit("c", {"status": "open", "review": {"state": "done"}}),
        ],
        max_distinct_values=2,
        min_units=2,
    )

    assert "| review.state | 3 | 2 | string (3) | max (2); pinboard (1) | draft; done |" in text
    assert "| status | 3 | 2 | string (3) | max (2); pinboard (1) | open; closed |" in text
    assert text.index("| review.state |") < text.index("| status |")


def test_metadata_enum_candidates_ignores_non_scalar_values_and_applies_filters():
    text = export_metadata_enum_candidates_markdown(
        [
            unit("a", {"kind": "note", "labels": ["one"], "owner": {"id": 1}}),
            unit("b", {"kind": "task", "labels": ["two"], "owner": {"id": 2}}),
            unit("c", {"kind": "event", "labels": ["three"], "owner": {"id": 3}}),
        ],
        max_distinct_values=2,
        min_units=2,
    )

    assert "labels" not in text
    assert "kind" not in text
    assert "| owner.id | 3 |" not in text
    assert "| _None_ | 0 | 0 | _None_ | _None_ | _None_ |" in text


def test_metadata_enum_candidates_writes_same_markdown_and_limits_examples(tmp_path):
    path = tmp_path / "reports" / "metadata-enums.md"
    units = [
        unit("a", {"state": "open"}),
        unit("b", {"state": "closed"}),
        unit("c", {"state": "blocked"}),
    ]

    text = export_metadata_enum_candidates_markdown(units, max_distinct_values=3, max_examples=2)
    stats = export_metadata_enum_candidates_markdown(units, path, max_distinct_values=3, max_examples=2)

    assert path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "candidates_exported": 1,
        "max_distinct_values": 3,
        "min_units": 2,
        "max_examples": 2,
        "bytes_written": path.stat().st_size,
    }
    assert "| state | 3 | 3 | string (3) | max (3) | blocked; closed |" in text
    assert "open" not in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_distinct_values": 0}, "max_distinct_values must be a positive integer"),
        ({"min_units": 0}, "min_units must be a positive integer"),
        ({"max_examples": -1}, "max_examples must be a non-negative integer"),
    ],
)
def test_metadata_enum_candidates_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_metadata_enum_candidates_markdown([], **kwargs)


def test_metadata_enum_candidates_is_importable_from_graph_export():
    from graph.export import export_metadata_enum_candidates_markdown as imported

    assert imported is export_metadata_enum_candidates_markdown
