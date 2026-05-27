from graph.store import summarize_unit_math_blocks


def test_inline_display_and_fence_counts():
    summary = summarize_unit_math_blocks([{"content": "Inline $x$.\n$$\ny\n$$\n```math\nz\n```"}])
    assert summary["inline_math_count"] == 1
    assert summary["display_math_count"] == 1
    assert summary["math_fence_count"] == 1
    assert summary["units_with_math"] == 1


def test_escaped_dollars_not_inline():
    assert summarize_unit_math_blocks([{"content": r"Cost \$5"}])["inline_math_count"] == 0


def test_empty_input():
    assert summarize_unit_math_blocks([]) == {"total_units": 0, "units_with_math": 0, "inline_math_count": 0, "display_math_count": 0, "math_fence_count": 0}
