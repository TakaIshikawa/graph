from __future__ import annotations

from dataclasses import dataclass

from graph.store.unit_content_structure_summary import summarize_unit_content_structure


@dataclass
class Unit:
    content: str
    metadata: dict[str, str]


def test_summarize_unit_content_structure_counts_unit_indicators_and_average_headings():
    summary = summarize_unit_content_structure(
        [
            {
                "source_project": "docs",
                "content": "# Title\n## Detail\n- item\n| A | B |\n| - | - |\n```python\n# ignored\n- ignored\n| ignored |\n```",
            },
            {"source_project": "docs", "content": "Plain text\n1. step"},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "docs",
            "unit_count": 2,
            "heading_count": 2,
            "list_count": 2,
            "table_count": 2,
            "code_block_count": 1,
            "link_count": 0,
            "units_with_headings": 1,
            "units_with_lists": 2,
            "units_with_tables": 1,
            "units_with_code_blocks": 1,
            "units_with_links": 0,
            "average_heading_count": "1.00",
        }
    ]


def test_summarize_unit_content_structure_supports_objects_and_unstructured_sources():
    summary = summarize_unit_content_structure(
        [
            Unit(content="No markdown", metadata={"source": "Beta"}),
            {"content": "Also plain", "metadata": {"source": "alpha"}},
        ]
    )

    assert [row["source"] for row in summary["rows"]] == ["alpha", "Beta"]
    assert summary["rows"][0]["unit_count"] == 1
    assert summary["rows"][0]["average_heading_count"] == "0.00"
    assert summary["rows"][1]["units_with_headings"] == 0
