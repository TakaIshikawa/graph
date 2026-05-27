from __future__ import annotations

from graph.store import summarize_unit_pandoc_citation_keys


def test_pandoc_citation_summary_counts_clusters_and_keys():
    report = summarize_unit_pandoc_citation_keys([{"id": "a", "content": "See [@doe2024; -@smith2020] and @doe2024."}])

    assert report["total_citation_mentions"] == 3
    assert report["unique_citation_keys"] == 2
    assert report["citation_cluster_count"] == 1
    assert report["suppress_author_count"] == 1
    assert report["top_citation_keys"][0]["key"] == "doe2024"


def test_pandoc_citation_summary_counts_multiple_keys_in_cluster():
    report = summarize_unit_pandoc_citation_keys([{"content": "[see @a; @b]"}])

    assert report["total_citation_mentions"] == 2


def test_pandoc_citation_summary_ignores_fenced_code():
    report = summarize_unit_pandoc_citation_keys([{"content": "```\n[@skip]\n```\n[@keep]"}])

    assert report["total_citation_mentions"] == 1
    assert report["top_citation_keys"][0]["key"] == "keep"
