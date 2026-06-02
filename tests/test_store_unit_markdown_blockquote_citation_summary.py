from graph.store.unit_markdown_blockquote_citation_summary import summarize_unit_markdown_blockquote_citations


def test_classifies_blockquote_citation_styles_and_ignores_regular_prose():
    result = summarize_unit_markdown_blockquote_citations(
        [
            {"id": "b", "content": "> Regular quoted prose\n> Source: Book"},
            {"id": "a", "content": "> -- Ada\n> — Grace\nNormal -- no"},
        ]
    )

    assert result == {
        "total_units": 2,
        "units_with_citations": 2,
        "citation_count": 3,
        "styles": {"dash": 1, "em_dash": 1, "source": 1},
        "samples": [
            {"unit_id": "b", "line_number": 2, "style": "source", "citation_text": "Book"},
            {"unit_id": "a", "line_number": 1, "style": "dash", "citation_text": "Ada"},
            {"unit_id": "a", "line_number": 2, "style": "em_dash", "citation_text": "Grace"},
        ],
        "units": [
            {"unit_id": "a", "citation_count": 2, "styles": {"dash": 1, "em_dash": 1}},
            {"unit_id": "b", "citation_count": 1, "styles": {"source": 1}},
        ],
    }
