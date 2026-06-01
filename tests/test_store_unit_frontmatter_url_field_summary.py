from __future__ import annotations

from graph.store import summarize_unit_frontmatter_url_fields


class Unit:
    def __init__(self, unit_id: str, metadata: dict[str, object] | None = None, frontmatter: dict[str, object] | None = None):
        self.id = unit_id
        self.metadata = metadata or {}
        self.frontmatter = frontmatter or {}


def test_unit_frontmatter_url_field_summary_reads_metadata_and_frontmatter_mappings():
    summary = summarize_unit_frontmatter_url_fields(
        [
            {"id": "u1", "metadata": {"canonical_url": "https://example.com/a", "title": "A"}},
            Unit("u2", metadata={"frontmatter": {"source_link": "docs/page", "author": "Ada"}}),
            Unit("u3", frontmatter={"source_uri": "urn:isbn:1234567890"}),
        ]
    )

    assert summary["url_field_counts"] == {"canonical_url": 1, "source_link": 1, "source_uri": 1}
    assert summary["scheme_counts"] == {"https": 1, "urn": 1}
    assert summary["missing_scheme_count"] == 1
    assert summary["field_samples"]["source_link"] == [{"unit_id": "u2", "value": "docs/page"}]


def test_unit_frontmatter_url_field_summary_supports_custom_field_names():
    summary = summarize_unit_frontmatter_url_fields(
        [
            {"id": "u1", "metadata": {"homepage": "https://example.com", "source": "https://ignored.example"}},
            {"id": "u2", "metadata": {"homepage": "example.org/about"}},
        ],
        field_names=["homepage"],
    )

    assert summary["url_field_counts"] == {"homepage": 2}
    assert summary["scheme_counts"] == {"https": 1}
    assert summary["missing_scheme_count"] == 1
    assert "source" not in summary["field_samples"]


def test_unit_frontmatter_url_field_summary_counts_schemes_and_invalid_values():
    summary = summarize_unit_frontmatter_url_fields(
        [
            {"id": "u1", "metadata": {"url": "https://example.com"}},
            {"id": "u2", "metadata": {"source_url": "ftp://files.example.com/archive"}},
            {"id": "u3", "metadata": {"link": "http:///missing-host"}},
            {"id": "u4", "metadata": {"canonical": "relative/path"}},
        ],
        sample_limit=1,
    )

    assert summary["scheme_counts"] == {"ftp": 1, "https": 1}
    assert summary["missing_scheme_count"] == 1
    assert summary["invalid_url_samples"] == [{"unit_id": "u3", "field": "link", "value": "http:///missing-host"}]
    assert summary["field_samples"]["url"] == [{"unit_id": "u1", "value": "https://example.com"}]
