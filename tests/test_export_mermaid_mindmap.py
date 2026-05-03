from __future__ import annotations

import pytest

from graph.export import export_units_to_mermaid_mindmap
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    *,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
    source_id: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
    )


def test_mermaid_mindmap_groups_units_under_sorted_tag_nodes():
    text = export_units_to_mermaid_mindmap(
        [
            unit("charlie", "Charlie", tags=["storage"]),
            unit("alpha", "Alpha", tags=["solar", "storage"]),
            unit("beta", "Beta", tags=["solar"]),
        ]
    )

    assert text == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    tag_0["solar"]\n'
        '      tag_0_unit_0["Alpha"]\n'
        '      tag_0_unit_1["Beta"]\n'
        '    tag_1["storage"]\n'
        '      tag_1_unit_0["Alpha"]\n'
        '      tag_1_unit_1["Charlie"]\n'
    )


def test_mermaid_mindmap_can_include_or_omit_untagged_units():
    units = [
        unit("tagged", "Tagged", tags=["topic"]),
        unit("untagged", "Untagged Title", tags=[]),
    ]

    omitted = export_units_to_mermaid_mindmap(units)
    included = export_units_to_mermaid_mindmap(units, include_untagged=True)

    assert "Untagged Title" not in omitted
    assert '    tag_0["topic"]' in omitted
    assert included == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    tag_0["topic"]\n'
        '      tag_0_unit_0["Tagged"]\n'
        '    tag_1["Untagged"]\n'
        '      tag_1_unit_0["Untagged Title"]\n'
    )


def test_mermaid_mindmap_deduplicates_tags_per_unit_and_limits_units_per_tag():
    text = export_units_to_mermaid_mindmap(
        [
            unit("beta", "Beta", tags=["topic", "topic"]),
            unit("alpha", "Alpha", tags=["topic"]),
            unit("gamma", "Gamma", tags=["topic"]),
        ],
        max_units_per_tag=2,
    )

    assert text == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    tag_0["topic"]\n'
        '      tag_0_unit_0["Alpha"]\n'
        '      tag_0_unit_1["Beta"]\n'
    )


def test_mermaid_mindmap_escapes_special_characters_in_labels():
    text = export_units_to_mermaid_mindmap(
        [
            unit(
                "special",
                'A [bracket] "quote" (paren) {brace} <tag> & slash \\ `tick`',
                tags=['topic [x] "q" (r) {s} <t> & slash \\ `tick`'],
            )
        ],
        root_label='Root [x] "q"',
    )

    assert text == (
        "mindmap\n"
        '  root["Root &#91;x&#93; &quot;q&quot;"]\n'
        '    tag_0["topic &#91;x&#93; &quot;q&quot; &#40;r&#41; &#123;s&#125; &lt;t&gt; &amp; slash &#92; &#96;tick&#96;"]\n'
        '      tag_0_unit_0["A &#91;bracket&#93; &quot;quote&quot; &#40;paren&#41; &#123;brace&#125; &lt;tag&gt; &amp; slash &#92; &#96;tick&#96;"]\n'
    )


def test_mermaid_mindmap_handles_empty_input():
    assert export_units_to_mermaid_mindmap([]) == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
    )


def test_mermaid_mindmap_can_group_by_source_collection_metadata():
    text = export_units_to_mermaid_mindmap(
        [
            unit(
                "beta",
                "Beta",
                metadata={"collection": {"name": "Research"}},
            ),
            unit(
                "alpha",
                "Alpha",
                metadata={"collections": ["Archive", {"name": "Research"}]},
            ),
        ],
        group_by="source_collection",
    )

    assert text == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    source_0["Archive"]\n'
        '      source_0_unit_0["Alpha"]\n'
        '    source_1["Research"]\n'
        '      source_1_unit_0["Alpha"]\n'
        '      source_1_unit_1["Beta"]\n'
    )


def test_mermaid_mindmap_source_collection_grouping_falls_back_to_source_type():
    text = export_units_to_mermaid_mindmap(
        [
            unit(
                "alpha",
                "Alpha",
                source_project="obsidian",
                source_entity_type="markdown",
            )
        ],
        group_by="source",
    )

    assert text == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    source_0["obsidian/markdown"]\n'
        '      source_0_unit_0["Alpha"]\n'
    )


def test_mermaid_mindmap_can_include_source_link_click_directives():
    text = export_units_to_mermaid_mindmap(
        [
            unit(
                "alpha",
                "Alpha",
                tags=["topic"],
                metadata={"source_url": 'https://example.com/a?quote="yes"'},
            ),
            unit("beta", "Beta", tags=["topic"], source_id="url:https://example.com/b"),
            unit("gamma", "Gamma", tags=["topic"]),
        ],
        include_source_links=True,
    )

    assert text == (
        "mindmap\n"
        '  root["Knowledge Units"]\n'
        '    tag_0["topic"]\n'
        '      tag_0_unit_0["Alpha"]\n'
        '      tag_0_unit_1["Beta"]\n'
        '      tag_0_unit_2["Gamma"]\n'
        'click tag_0_unit_0 "https://example.com/a?quote=%22yes%22" "Open source"\n'
        'click tag_0_unit_1 "https://example.com/b" "Open source"\n'
    )


def test_mermaid_mindmap_output_is_deterministic():
    units = [
        unit("beta", "Same", tags=["zeta", "alpha"]),
        unit("alpha", "Same", tags=["alpha"]),
        unit("gamma", "Gamma", tags=["zeta"]),
    ]

    first = export_units_to_mermaid_mindmap(units)
    second = export_units_to_mermaid_mindmap(reversed(units))

    assert first == second


def test_mermaid_mindmap_is_importable_from_graph_export():
    from graph.export import export_units_to_mermaid_mindmap as imported

    assert imported is export_units_to_mermaid_mindmap


@pytest.mark.parametrize("max_units_per_tag", [-1, 1.5, True])
def test_mermaid_mindmap_validates_max_units_per_tag(max_units_per_tag):
    with pytest.raises(
        ValueError, match="max_units_per_tag must be a non-negative integer or None"
    ):
        export_units_to_mermaid_mindmap([], max_units_per_tag=max_units_per_tag)


def test_mermaid_mindmap_validates_include_untagged():
    with pytest.raises(ValueError, match="include_untagged must be a boolean"):
        export_units_to_mermaid_mindmap([], include_untagged="yes")


def test_mermaid_mindmap_validates_group_by():
    with pytest.raises(ValueError, match="group_by must be 'tag' or 'source_collection'"):
        export_units_to_mermaid_mindmap([], group_by="topic")


def test_mermaid_mindmap_validates_include_source_links():
    with pytest.raises(ValueError, match="include_source_links must be a boolean"):
        export_units_to_mermaid_mindmap([], include_source_links="yes")
