from __future__ import annotations

from graph.store.source_generator_meta_summary import summarize_source_generator_meta


def test_source_generator_meta_summary_counts_keys_and_html_meta():
    summary = summarize_source_generator_meta(
        [
            {"source_id": "a", "generator": " WordPress 6.5 "},
            {"source_id": "b", "metadata": {"meta_generator": "Hugo 0.124"}},
            {"source_id": "c", "content": '<meta name="generator" content="Next.js">'},
            {"source_id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_generator"] == 3
    assert summary["generator_counts"] == {"Hugo 0.124": 1, "Next.js": 1, "WordPress 6.5": 1}
    assert summary["wordpress_count"] == 1
    assert summary["static_site_generator_count"] == 2
    assert summary["missing_generator_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "generator": "WordPress 6.5"},
        {"source_id": "b", "generator": "Hugo 0.124"},
        {"source_id": "c", "generator": "Next.js"},
    ]


def test_source_generator_meta_summary_bounds_samples_deterministically():
    summary = summarize_source_generator_meta(
        [
            {"source_id": "b", "generator": "Jekyll"},
            {"source_id": "a", "generator": "Gatsby"},
        ],
        sample_limit=1,
    )

    assert summary["samples"] == [{"source_id": "a", "generator": "Gatsby"}]
