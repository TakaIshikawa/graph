from graph.store import summarize_source_sdk_languages


def test_sdk_language_summary_counts_languages_and_package_hints():
    summary = summarize_source_sdk_languages(
        [
            {"source_id": "a", "url": "https://github.com/acme/python-sdk", "content": "Python SDK pip install acme"},
            {"source_id": "b", "content": "Client library for JavaScript and TypeScript via npm install acme."},
            {"source_id": "c", "content": "We use Java internally but publish no SDK."},
        ]
    )

    assert summary["sources_with_sdk_language_hints"] == 3
    assert summary["language_counts"]["python"] == 1
    assert summary["language_counts"]["javascript"] == 1
    assert summary["language_counts"]["typescript"] == 1
    assert summary["language_counts"]["java"] == 1
    assert summary["package_hint_counts"] == {"github": 1, "npm": 1, "pypi": 1}
