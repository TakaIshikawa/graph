from __future__ import annotations

import pytest

from graph.export import export_tag_cooccurrence_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_export_tag_cooccurrence_markdown_writes_summary_and_stable_ordering():
    units = [
        unit("unit-b", ["gamma", "alpha"]),
        unit("unit-a", ["beta", "alpha"]),
        unit("unit-c", ["beta", "alpha"]),
    ]

    first = export_tag_cooccurrence_markdown(units)
    second = export_tag_cooccurrence_markdown(reversed(units))

    assert first == second
    assert first == (
        "# Tag Co-occurrence\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Option | Value |\n"
        "| --- | ---: |\n"
        "| Units scanned | 3 |\n"
        "| Tags found | 3 |\n"
        "| Pairs reported | 2 |\n"
        "| Min count | 1 |\n"
        "\n"
        "## Tag Pairs\n"
        "\n"
        "| Tag A | Tag B | Count |\n"
        "| --- | --- | ---: |\n"
        "| alpha | beta | 2 |\n"
        "| alpha | gamma | 1 |\n"
    )


def test_export_tag_cooccurrence_markdown_filters_and_limits_after_sorting():
    text = export_tag_cooccurrence_markdown(
        [
            unit("unit-a", ["alpha", "beta", "gamma"]),
            unit("unit-b", ["alpha", "beta"]),
            unit("unit-c", ["alpha", "gamma"]),
            unit("unit-d", ["beta", "gamma"]),
        ],
        min_count=2,
        limit=2,
    )

    assert "| Min count | 2 |" in text
    assert "| Limit | 2 |" in text
    assert "| Pairs reported | 2 |" in text
    assert "| alpha | beta | 2 |\n| alpha | gamma | 2 |" in text
    assert "| beta | gamma | 2 |" not in text


def test_export_tag_cooccurrence_markdown_escapes_table_cells():
    text = export_tag_cooccurrence_markdown([unit("unit-a", [r"alpha|beta", r"path\\tag"])])

    assert r"| alpha\|beta | path\\\\tag | 1 |" in text


def test_export_tag_cooccurrence_markdown_returns_empty_report_for_empty_input():
    assert export_tag_cooccurrence_markdown([]) == (
        "# Tag Co-occurrence\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Option | Value |\n"
        "| --- | ---: |\n"
        "| Units scanned | 0 |\n"
        "| Tags found | 0 |\n"
        "| Pairs reported | 0 |\n"
        "| Min count | 1 |\n"
        "\n"
        "## Tag Pairs\n"
        "\n"
        "| Tag A | Tag B | Count |\n"
        "| --- | --- | ---: |\n"
        "| _None_ | _None_ | 0 |\n"
    )


@pytest.mark.parametrize("min_count", [0, -1, "2", None, True])
def test_export_tag_cooccurrence_markdown_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_tag_cooccurrence_markdown([], min_count=min_count)


@pytest.mark.parametrize("limit", [-1, "2", True])
def test_export_tag_cooccurrence_markdown_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        export_tag_cooccurrence_markdown([], limit=limit)


def test_export_tag_cooccurrence_markdown_writes_path_and_returns_stats(tmp_path):
    output_path = tmp_path / "reports" / "tags.md"

    stats = export_tag_cooccurrence_markdown(
        [unit("unit-a", ["alpha", "beta"]), unit("unit-b", ["alpha", "beta"])],
        output_path,
        min_count=2,
        limit=1,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "| alpha | beta | 2 |" in text
    assert stats == {
        "path": str(output_path),
        "rows_exported": 1,
        "units_scanned": 2,
        "min_count": 2,
        "limit": 1,
        "bytes_written": output_path.stat().st_size,
    }
