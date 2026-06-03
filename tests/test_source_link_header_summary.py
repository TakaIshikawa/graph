from graph.store.source_link_header_summary import summarize_source_link_headers


def test_link_summary_parses_multiple_links():
    summary = summarize_source_link_headers(
        [
            {
                "source_id": "a",
                "Link": '<https://example.com/>; rel=canonical, <https://example.com/feed>; rel=alternate',
            }
        ]
    )

    assert summary["total_sources"] == 1
    assert summary["sources_with_link_header"] == 1
    assert summary["missing_link_header_count"] == 0
    assert summary["rel_counts"] == {"alternate": 1, "canonical": 1}


def test_link_summary_splits_quoted_rel_lists():
    summary = summarize_source_link_headers(
        [
            {
                "source_id": "a",
                "Link": '<https://example.com/feed>; rel="alternate preload"; as=fetch',
            }
        ]
    )

    assert summary["rel_counts"] == {"alternate": 1, "preload": 1}
    assert summary["rows"] == [
        {
            "rel": "alternate",
            "count": 1,
            "source_ids": ["a"],
            "examples": ['<https://example.com/feed>; rel="alternate preload"; as=fetch'],
        },
        {
            "rel": "preload",
            "count": 1,
            "source_ids": ["a"],
            "examples": ['<https://example.com/feed>; rel="alternate preload"; as=fetch'],
        },
    ]


def test_link_summary_counts_canonical_preload_and_preconnect_relations():
    summary = summarize_source_link_headers(
        [
            {"source_id": "canonical", "Link": "<https://example.com/>; rel=canonical"},
            {"source_id": "preload", "Link": "</app.css>; rel=preload"},
            {"source_id": "preconnect", "Link": "<https://cdn.example.com>; rel=preconnect"},
        ]
    )

    assert summary["rel_counts"] == {"canonical": 1, "preconnect": 1, "preload": 1}


def test_link_summary_ignores_malformed_rels_and_keeps_valid_entries():
    summary = summarize_source_link_headers(
        [
            {
                "source_id": "mixed",
                "Link": '<https://bad>; rel="bad,rel", <https://good>; rel=alternate, ; rel=canonical',
            }
        ]
    )

    assert summary["sources_with_link_header"] == 1
    assert summary["rel_counts"] == {"alternate": 1, "canonical": 1}


def test_link_summary_reads_metadata_and_nested_headers():
    summary = summarize_source_link_headers(
        [
            {"source_id": "direct", "link": "</a>; rel=canonical"},
            {"source_id": "nested", "headers": {"LINK": "</b>; rel=alternate"}},
            {"source_id": "metadata", "metadata": {"response_headers": {"link": "</c>; rel=preload"}}},
            {"source_id": "missing"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_link_header"] == 3
    assert summary["missing_link_header_count"] == 1
    assert summary["rel_counts"] == {"alternate": 1, "canonical": 1, "preload": 1}


def test_link_summary_bounds_row_source_ids_and_examples():
    summary = summarize_source_link_headers(
        [
            {"source_id": "a", "Link": "</a>; rel=canonical"},
            {"source_id": "b", "Link": "</b>; rel=canonical"},
            {"source_id": "c", "Link": "</c>; rel=canonical"},
        ],
        sample_limit=2,
    )

    assert summary["rows"] == [
        {
            "rel": "canonical",
            "count": 3,
            "source_ids": ["a", "b"],
            "examples": ["</a>; rel=canonical", "</b>; rel=canonical"],
        }
    ]
