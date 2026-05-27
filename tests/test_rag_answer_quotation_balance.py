from __future__ import annotations

from graph.rag.answer_quotation_balance import audit_answer_quotation_balance


def test_answer_quotation_balance_counts_blockquotes_and_inline_quotes():
    audit = audit_answer_quotation_balance('Intro words here.\n> quoted block text\nThen "inline quote" ends.')

    assert audit["quote_count"] == 2
    assert audit["quoted_word_count"] == 5
    assert audit["total_word_count"] == 10
    assert audit["quote_word_ratio"] == 0.5
    assert audit["balance_flags"] == []


def test_answer_quotation_balance_flags_long_and_quote_heavy_answers():
    long_quote = "> " + " ".join(f"word{i}" for i in range(30))
    audit = audit_answer_quotation_balance(f"{long_quote}\nshort note")

    assert audit["long_quote_count"] == 1
    assert audit["balance_flags"] == ["long_quote", "quote_heavy"]


def test_answer_quotation_balance_flags_answers_without_quotes():
    audit = audit_answer_quotation_balance("Plain answer with no quoted text.")

    assert audit["quote_count"] == 0
    assert audit["quote_word_ratio"] == 0.0
    assert audit["balance_flags"] == ["no_quotes"]
