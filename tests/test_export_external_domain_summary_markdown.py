from __future__ import annotations

import pytest

from graph.export import export_external_domain_summary_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    title: str | None = None,
    metadata: dict | None = None,
    content: str = "",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Unit {unit_id}",
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def test_external_domain_summary_counts_metadata_content_and_repeated_urls_once_per_unit():
    text = export_external_domain_summary_markdown(
        [
            unit(
                "b",
                source_project=SourceProject.PRESENCE,
                metadata={"links": ["https://Example.com/b", {"url": "https://EXAMPLE.com/again"}]},
                content="Read https://docs.example.com/guide and https://example.com/content.",
            ),
            unit(
                "a",
                metadata={"source": {"url": "HTTPS://Example.COM/a"}},
                content="Repeated https://example.com/a https://example.com/a.",
            ),
        ]
    )

    assert "| example.com | 2 | max (1); presence (1) | content (2); links (1); source.url (1) | Unit a; Unit b |" in text
    assert "| docs.example.com | 1 | presence (1) | content (1) | Unit b |" in text
    assert text.index("| example.com |") < text.index("| docs.example.com |")


def test_external_domain_summary_filters_metadata_keys_but_still_counts_content_urls():
    text = export_external_domain_summary_markdown(
        [
            unit(
                "a",
                metadata={
                    "source": {"url": "https://alpha.test/a"},
                    "canonical_url": "https://beta.test/b",
                },
                content="https://content.test/c",
            ),
            unit("b", metadata={"source": {"url": "https://alpha.test/b"}}),
        ],
        metadata_keys=["source.url"],
    )

    assert "| alpha.test | 2 | max (2) | source.url (2) | Unit a; Unit b |" in text
    assert "| content.test | 1 | max (1) | content (1) | Unit a |" in text
    assert "beta.test" not in text
    assert "| Metadata keys | source.url |" in text


def test_external_domain_summary_applies_min_count_limit_and_writes_same_markdown(tmp_path):
    path = tmp_path / "reports" / "domains.md"
    units = [
        unit("a", metadata={"url": "https://alpha.test/a"}),
        unit("b", metadata={"url": "https://alpha.test/b"}),
        unit("c", metadata={"url": "https://beta.test/c"}),
    ]

    text = export_external_domain_summary_markdown(units, min_unit_count=2, limit=1)
    stats = export_external_domain_summary_markdown(units, path, min_unit_count=2, limit=1)

    assert path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "domains_exported": 1,
        "min_unit_count": 2,
        "limit": 1,
        "metadata_keys": None,
        "bytes_written": path.stat().st_size,
    }
    assert "| alpha.test | 2 |" in text
    assert "beta.test" not in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metadata_keys": "url"}, "metadata_keys must be a sequence"),
        ({"metadata_keys": [""]}, "metadata_keys must be a sequence"),
        ({"min_unit_count": 0}, "min_unit_count must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer or None"),
    ],
)
def test_external_domain_summary_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_external_domain_summary_markdown([], **kwargs)


def test_external_domain_summary_is_importable_from_graph_export():
    from graph.export import export_external_domain_summary_markdown as imported

    assert imported is export_external_domain_summary_markdown
