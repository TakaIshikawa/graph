from __future__ import annotations

import hashlib

from graph.store.unit_content_checksum_drift_summary import summarize_unit_content_checksum_drift


def test_unit_content_checksum_drift_classifies_sha256_md5_and_stale():
    summary = summarize_unit_content_checksum_drift(
        [
            {"id": "u1", "content": "alpha", "metadata": {"sha256": hashlib.sha256(b"alpha").hexdigest()}},
            {"id": "u2", "content": "beta", "metadata": {"content_md5": hashlib.md5(b"beta").hexdigest()}},
            {"id": "u3", "content": "changed", "metadata": {"checksum": "sha256:" + hashlib.sha256(b"old").hexdigest()}},
        ]
    )

    assert summary["counts"]["matching"] == 2
    assert summary["counts"]["stale"] == 1
    assert summary["stale_examples"][0]["unit_id"] == "u3"


def test_unit_content_checksum_drift_handles_missing_and_unknown_formats():
    summary = summarize_unit_content_checksum_drift(
        [{"id": "u1", "content": "x"}, {"id": "u2"}, {"id": "u3", "content": "x", "metadata": {"checksum": "weird"}}]
    )

    assert summary["counts"] == {"matching": 0, "missing": 1, "stale": 0, "algorithm_unknown": 1, "missing_content": 1}
