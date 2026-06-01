from graph.store import summarize_source_changelog_hints


def test_changelog_summary_counts_cues_and_urls():
    summary = summarize_source_changelog_hints(
        [
            {"source_id": "a", "content": "Changelog https://example.com/changelog and release notes."},
            {"source_id": "b", "metadata": {"body": "Migration guide with breaking changes and deprecation notice."}},
            {"source_id": "c", "content": "Current docs."},
        ]
    )

    assert summary["sources_with_changelog_hints"] == 2
    assert summary["cue_counts"] == {"breaking_change": 1, "changelog": 1, "deprecation": 1, "migration": 1, "release_note": 1}
    assert summary["changelog_url_samples"] == ["https://example.com/changelog"]
    assert summary["samples"][0]["source_id"] == "a"
