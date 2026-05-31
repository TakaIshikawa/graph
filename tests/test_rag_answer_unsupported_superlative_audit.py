from graph.rag.answer_unsupported_superlative import audit_answer_unsupported_superlatives


def test_cited_superlatives_are_tracked_separately():
    result = audit_answer_unsupported_superlatives("It is the fastest option according to the benchmark [1].")

    assert result["unsupported_count"] == 0
    assert result["cited_count"] == 1


def test_uncited_superlatives_are_flagged():
    result = audit_answer_unsupported_superlatives("This is the best and only viable option.")

    assert [claim["phrase"] for claim in result["flagged_claims"]] == ["best", "only"]


def test_ordinary_comparatives_are_ignored():
    result = audit_answer_unsupported_superlatives("This option is faster and better supported than the old one.")

    assert result["flagged_claims"] == []


def test_quoted_source_text_is_ignored():
    result = audit_answer_unsupported_superlatives(' "This is the largest trial." The answer summarizes it cautiously.')

    assert result["unsupported_count"] == 0


def test_multiple_flagged_claims_are_reported():
    result = audit_answer_unsupported_superlatives("It is the first product. It is also the strongest choice.")

    assert [(claim["sentence_index"], claim["phrase"]) for claim in result["flagged_claims"]] == [(0, "first"), (1, "strongest")]
