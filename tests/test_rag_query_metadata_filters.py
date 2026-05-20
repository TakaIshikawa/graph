from __future__ import annotations

from graph.rag.query_metadata_filters import extract_query_metadata_filter_hints


def test_query_metadata_filters_empty_input_returns_stable_structure():
    assert extract_query_metadata_filter_hints("  \n ") == {
        "filters": {
            "source": [],
            "tag": [],
            "author": [],
            "project": [],
            "account": [],
        },
        "quoted_hints": [],
        "terms": [],
        "ignored_tokens": [],
    }


def test_query_metadata_filters_parse_multiple_filters_and_terms():
    payload = extract_query_metadata_filter_hints(
        "source:github tag:python tag:rag project:Search account:Acme deployment notes"
    )

    assert payload["filters"] == {
        "source": ["github"],
        "tag": ["python", "rag"],
        "author": [],
        "project": ["Search"],
        "account": ["Acme"],
    }
    assert payload["terms"] == ["deployment", "notes"]
    assert payload["quoted_hints"] == []
    assert payload["ignored_tokens"] == []


def test_query_metadata_filters_preserve_quoted_values_with_spaces():
    payload = extract_query_metadata_filter_hints(
        'author:"Jane Doe" project:"Knowledge Graph" account:`Enterprise Search` "Roadmap Q2"'
    )

    assert payload["filters"]["author"] == ["Jane Doe"]
    assert payload["filters"]["project"] == ["Knowledge Graph"]
    assert payload["filters"]["account"] == ["Enterprise Search"]
    assert payload["quoted_hints"] == ["Roadmap Q2"]


def test_query_metadata_filters_report_malformed_filters_without_raising():
    payload = extract_query_metadata_filter_hints('source: tag:"" owner:ada bad:value:shape account:')

    assert payload["filters"] == {
        "source": [],
        "tag": [],
        "author": [],
        "project": [],
        "account": [],
    }
    assert payload["ignored_tokens"] == ["source:", 'tag:""', "owner:ada", "bad:value:shape", "account:"]
