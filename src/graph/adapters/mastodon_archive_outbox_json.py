"""Compatibility adapter for Mastodon archive outbox JSON exports."""

from __future__ import annotations

from graph.adapters.mastodon_outbox_json import MastodonOutboxJsonAdapter


class MastodonArchiveOutboxJsonAdapter(MastodonOutboxJsonAdapter):
    @property
    def name(self) -> str:
        return "mastodon_archive_outbox_json"
