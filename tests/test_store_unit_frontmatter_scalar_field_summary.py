from __future__ import annotations

from graph.store import summarize_unit_frontmatter_scalar_fields


def test_summarize_unit_frontmatter_scalar_fields_returns_deterministic_rows():
    summary = summarize_unit_frontmatter_scalar_fields(
        [
            {"id": "a", "content": "---\ntitle: Alpha\npublished: true\nrating: 4.5\n---\nBody"},
            {"id": "b", "content": "---\ntitle: Beta\npublished: false\ncreated: 2024-01-02\n---"},
            {"id": "c", "content": "title: Not frontmatter"},
        ]
    )

    assert summary == {
        "total_units": 3,
        "units_with_frontmatter": 2,
        "scalar_fields": [
            {
                "key_path": "created",
                "unit_count": 1,
                "blank_value_count": 0,
                "most_common_type_hint": "date-like",
                "example_values": ["2024-01-02"],
            },
            {
                "key_path": "published",
                "unit_count": 2,
                "blank_value_count": 0,
                "most_common_type_hint": "boolean-like",
                "example_values": ["true", "false"],
            },
            {
                "key_path": "rating",
                "unit_count": 1,
                "blank_value_count": 0,
                "most_common_type_hint": "numeric-like",
                "example_values": ["4.5"],
            },
            {
                "key_path": "title",
                "unit_count": 2,
                "blank_value_count": 0,
                "most_common_type_hint": "string",
                "example_values": ["Alpha", "Beta"],
            },
        ],
    }


def test_summarize_unit_frontmatter_scalar_fields_types_blanks_and_quotes():
    summary = summarize_unit_frontmatter_scalar_fields(
        [
            {
                "content": "\n".join(
                    [
                        "---",
                        "blank:",
                        "quoted: 'yes'",
                        'also_quoted: "42"',
                        "enabled: on",
                        "---",
                    ]
                )
            }
        ]
    )

    hints = {row["key_path"]: row for row in summary["scalar_fields"]}
    assert hints["blank"]["blank_value_count"] == 1
    assert hints["blank"]["most_common_type_hint"] == "blank"
    assert hints["quoted"]["most_common_type_hint"] == "quoted"
    assert hints["quoted"]["example_values"] == ["yes"]
    assert hints["also_quoted"]["most_common_type_hint"] == "quoted"
    assert hints["enabled"]["most_common_type_hint"] == "boolean-like"


def test_summarize_unit_frontmatter_scalar_fields_handles_simple_indented_paths_and_skips_collections():
    summary = summarize_unit_frontmatter_scalar_fields(
        [
            {
                "content": "\n".join(
                    [
                        "---",
                        "author:",
                        "  name: Ada",
                        "  links:",
                        "    - https://example.com",
                        "tags: [one, two]",
                        "notes: |",
                        "  skipped",
                        "---",
                    ]
                )
            }
        ]
    )

    assert summary["scalar_fields"] == [
        {
            "key_path": "author.name",
            "unit_count": 1,
            "blank_value_count": 0,
            "most_common_type_hint": "string",
            "example_values": ["Ada"],
        }
    ]
