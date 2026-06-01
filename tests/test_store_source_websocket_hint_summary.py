from __future__ import annotations

from graph.store.source_websocket_hint_summary import summarize_source_websocket_hints


def test_source_websocket_hints_count_secure_websocket_urls():
    summary = summarize_source_websocket_hints(
        [
            {"id": "s1", "metadata": {"url": "wss://events.example.test/socket"}},
            {"id": "s2", "url": "ws://events.example.test/feed"},
        ]
    )

    assert summary["total_sources"] == 2
    assert summary["websocket_source_count"] == 2
    assert summary["secure_websocket_count"] == 1
    assert summary["realtime_hint_count"] == 0
    assert summary["samples"][0] == {
        "source_id": "s1",
        "field": "metadata.url",
        "hint_type": "secure_websocket",
        "value": "wss://events.example.test/socket",
    }


def test_source_websocket_hints_count_realtime_text_without_ws_url():
    summary = summarize_source_websocket_hints(
        [
            {"id": "s1", "description": "Realtime subscription feed for order updates."},
            {"id": "s2", "metadata": {"notes": "Streaming API via socket.io namespace."}},
        ]
    )

    assert summary["websocket_source_count"] == 1
    assert summary["secure_websocket_count"] == 0
    assert summary["realtime_hint_count"] == 2


def test_source_websocket_hints_ignore_unrelated_http_sources_and_limit_samples():
    summary = summarize_source_websocket_hints(
        [
            {"id": "s1", "url": "https://example.test/poll", "metadata": {"description": "Daily HTTP export"}},
            {"id": "s2", "metadata": {"url": "https://example.test/api"}},
        ],
        sample_limit=1,
    )

    assert summary == {
        "total_sources": 2,
        "websocket_source_count": 0,
        "secure_websocket_count": 0,
        "realtime_hint_count": 0,
        "samples": [],
    }
