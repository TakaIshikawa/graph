from graph.store.unit_markdown_ruby_text_summary import summarize_unit_markdown_ruby_text


def test_summarizes_valid_ruby_pairs_and_ignores_malformed_markup():
    result = summarize_unit_markdown_ruby_text(
        [
            {"id": "b", "content": "<ruby>漢<rt>kan</rt></ruby>"},
            {"id": "a", "content": "<ruby>字<rt>ji</rt></ruby> <ruby>bad</ruby>"},
        ]
    )

    assert result == {
        "total_units": 2,
        "units_with_ruby": 2,
        "ruby_count": 2,
        "annotation_count": 2,
        "samples": [
            {"unit_id": "a", "base_text": "字", "annotation_text": "ji"},
            {"unit_id": "b", "base_text": "漢", "annotation_text": "kan"},
        ],
        "units": [
            {"unit_id": "a", "ruby_count": 1, "annotation_count": 1},
            {"unit_id": "b", "ruby_count": 1, "annotation_count": 1},
        ],
    }
