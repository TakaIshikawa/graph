from __future__ import annotations

import pytest

from graph.export import export_units_to_mermaid_mindmap
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str, *, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata={},
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
