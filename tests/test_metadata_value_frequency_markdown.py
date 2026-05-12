from __future__ import annotations

import pytest

from graph.export import export_metadata_value_frequency_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="record",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_value_frequency_counts_scalars_and_list_members():
    text = export_metadata_value_frequency_markdown(
        [
            unit("a", {"status": "open", "labels": ["red", "blue"], "ignored": {"nested": "x"}}),
            unit("b", {"status": "closed", "labels": ["red", "green"]}),
            unit("c", {"status": "open", "labels": ["red", {"skip": True}, None]}),
        ],
        top_values=3,
    )

    assert "| Units scanned | 3 |" in text
    assert "| Keys reported | 2 |" in text
    assert "### labels" in text
    assert "| red | 3 |\n| null | 1 |\n| blue | 1 |" in text
    assert "### status" in text
    assert "| open | 2 |\n| closed | 1 |" in text
    assert "ignored" not in text


def test_metadata_value_frequency_filters_keys_and_escapes_cells_deterministically():
    units = [
        unit("b", {"status": "open|blocked", "kind": r"path\\value"}),
        unit("a", {"status": "open|blocked", "kind": "other"}),
        unit("c", {"status": "closed", "kind": r"path\\value"}),
    ]

    first = export_metadata_value_frequency_markdown(units, keys=["status"], top_values=2)
    second = export_metadata_value_frequency_markdown(reversed(units), keys=["status"], top_values=2)

    assert first == second
    assert "| Keys filter | status |" in first
    assert "### status" in first
    assert r"| open\|blocked | 2 |" in first
    assert "### kind" not in first
    assert r"path\\\\value" not in first


def test_metadata_value_frequency_applies_min_count_and_top_values():
    text = export_metadata_value_frequency_markdown(
        [
            unit("a", {"status": "open"}),
            unit("b", {"status": "open"}),
            unit("c", {"status": "closed"}),
            unit("d", {"status": "pending"}),
        ],
        top_values=1,
        min_count=2,
    )

    assert "| Top values | 1 |" in text
    assert "| Min count | 2 |" in text
    assert "| open | 2 |" in text
    assert "closed" not in text
    assert "pending" not in text


def test_metadata_value_frequency_writes_path_and_returns_stats(tmp_path):
    output_path = tmp_path / "reports" / "metadata-values.md"
    units = [unit("a", {"status": "open"})]

    text = export_metadata_value_frequency_markdown(units, keys=["status"])
    stats = export_metadata_value_frequency_markdown(units, output_path, keys=["status"])

    assert output_path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(output_path),
        "units_scanned": 1,
        "keys_requested": ["status"],
        "keys_exported": 1,
        "top_values": 5,
        "min_count": 1,
        "bytes_written": output_path.stat().st_size,
    }


@pytest.mark.parametrize("top_values", [0, -1, "2", None, True])
def test_metadata_value_frequency_validates_top_values(top_values):
    with pytest.raises(ValueError, match="top_values must be a positive integer"):
        export_metadata_value_frequency_markdown([], top_values=top_values)


@pytest.mark.parametrize("min_count", [0, -1, "2", None, True])
def test_metadata_value_frequency_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_metadata_value_frequency_markdown([], min_count=min_count)


def test_metadata_value_frequency_validates_keys():
    with pytest.raises(ValueError, match="keys must contain at least one non-empty key or be None"):
        export_metadata_value_frequency_markdown([], keys=["", "   "])


def test_metadata_value_frequency_is_importable_from_graph_export():
    from graph.export import export_metadata_value_frequency_markdown as imported

    assert imported is export_metadata_value_frequency_markdown
