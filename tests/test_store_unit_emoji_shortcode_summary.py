from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_unit_emoji_shortcodes


def test_summarize_unit_emoji_shortcodes_groups_usage_by_source():
    summary = summarize_unit_emoji_shortcodes(
        [
            {"source": "slack", "content": "Ship it :rocket: :white_check_mark:"},
            {"source": "slack", "content": "Again :rocket: and invalid :Rocket:"},
            SimpleNamespace(source="discord", content="Hello :wave:"),
            {"metadata": {"source": "slack"}, "content": "No shortcode"},
        ]
    )

    assert summary == {
        "total_units": 4,
        "sources": [
            {
                "source": "discord",
                "unit_count": 1,
                "units_with_shortcodes": 1,
                "shortcode_count": 1,
                "unique_shortcode_count": 1,
                "most_common_shortcode": "wave",
                "invalid_shortcode_count": 0,
            },
            {
                "source": "slack",
                "unit_count": 3,
                "units_with_shortcodes": 2,
                "shortcode_count": 3,
                "unique_shortcode_count": 2,
                "most_common_shortcode": "rocket",
                "invalid_shortcode_count": 1,
            },
        ],
    }


def test_summarize_unit_emoji_shortcodes_uses_deterministic_tie_breaking():
    summary = summarize_unit_emoji_shortcodes(
        [{"source": "chat", "content": ":zap: :apple: :zap: :apple:"}]
    )

    assert summary["sources"][0]["most_common_shortcode"] == "apple"


def test_summarize_unit_emoji_shortcodes_ignores_urls_and_fenced_code_blocks():
    summary = summarize_unit_emoji_shortcodes(
        [
            {
                "source": "chat",
                "content": "\n".join(
                    [
                        "Visible :ok_hand:",
                        "https://example.com/path/:not_emoji:/view",
                        "```",
                        "hidden :rocket:",
                        "```",
                    ]
                ),
            }
        ]
    )

    assert summary["sources"] == [
        {
            "source": "chat",
            "unit_count": 1,
            "units_with_shortcodes": 1,
            "shortcode_count": 1,
            "unique_shortcode_count": 1,
            "most_common_shortcode": "ok_hand",
            "invalid_shortcode_count": 0,
        }
    ]
