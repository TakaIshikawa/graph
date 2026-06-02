from graph.rag.result_citation_density import analyze_result_citation_density


def test_analyzes_mixed_cited_and_uncited_results():
    result = analyze_result_citation_density(
        [
            {"id": "a", "title": "Cited", "citations": ["src-1"], "text": "A"},
            {"id": "b", "title": "Sparse", "text": "B"},
            {"id": "c", "title": "URL cited", "source_url": "https://example.com/c"},
        ]
    )

    assert result["result_count"] == 3
    assert result["cited_result_count"] == 2
    assert result["uncited_result_count"] == 1
    assert result["citation_density"] == 0.6667
    assert result["sparse_results"] == [{"id": "b", "title": "Sparse"}]


def test_empty_input_has_zero_density():
    assert analyze_result_citation_density([]) == {
        "result_count": 0,
        "cited_result_count": 0,
        "uncited_result_count": 0,
        "citation_density": 0.0,
        "sparse_results": [],
    }


def test_recognizes_alternate_citation_field_names():
    result = analyze_result_citation_density(
        [
            {"result_id": "ref", "references": [{"url": "https://example.com"}]},
            {"result_id": "url", "url": "https://example.com/item"},
            {"result_id": "citation", "citation_id": "S1"},
        ]
    )

    assert result["cited_result_count"] == 3
    assert result["sparse_results"] == []
