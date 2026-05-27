from graph.store import summarize_unit_citation_keys


def test_single_bracketed_and_multi_citations():
    summary = summarize_unit_citation_keys([{"content": "@doe2024 and [@roe2023] and [@a2020; @b2021]"}])
    assert summary["total_citations"] == 4
    assert summary["unique_key_count"] == 4
    assert summary["units_with_citations"] == 1


def test_email_and_mentions_ignored():
    assert summarize_unit_citation_keys([{"content": "a@b.com @person"}])["total_citations"] == 0


def test_code_fences_ignored_and_empty_input():
    assert summarize_unit_citation_keys([{"content": "```\n@doe2024\n```"}])["total_citations"] == 0
    assert summarize_unit_citation_keys([]) == {"total_citations": 0, "unique_key_count": 0, "units_with_citations": 0, "most_common_keys": []}
