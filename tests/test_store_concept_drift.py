"""Tests for Store.detect_concept_drift."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _make_unit(
    source_id: str,
    title: str,
    content: str,
    created_at: datetime,
    source_project: str = SourceProject.FORTY_TWO,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="knowledge_node",
        title=title,
        content=content,
        content_type=ContentType.FINDING,
        created_at=created_at,
        updated_at=created_at,
    )


_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


class TestDetectConceptDrift:
    def test_empty_database(self, store: Store):
        drifts = store.detect_concept_drift(timedelta(days=7))
        assert drifts == []

    def test_single_window_no_drift(self, store: Store):
        """Only one window → nothing to compare → no drift."""
        store.insert_unit(
            _make_unit("n1", "Python basics", "Learn python programming", _BASE)
        )
        store.insert_unit(
            _make_unit("n2", "Python advanced", "Advanced python topics", _BASE + timedelta(hours=1))
        )
        drifts = store.detect_concept_drift(timedelta(days=30))
        assert drifts == []

    def test_identical_windows_no_drift(self, store: Store):
        """Identical content across windows → similarity ~1.0 → no drift."""
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w1_{i}",
                    "Machine learning",
                    "Neural networks deep learning models training",
                    _BASE + timedelta(hours=i),
                )
            )
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}",
                    "Machine learning",
                    "Neural networks deep learning models training",
                    _BASE + timedelta(days=8, hours=i),
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.1)
        # Content is identical → cosine sim ≈ 1.0 → drift ≈ 0.0 → no events
        assert drifts == []

    def test_drift_detected_with_different_content(self, store: Store):
        """Completely different content across windows → high drift."""
        # Window 1: all about cooking
        for i in range(5):
            store.insert_unit(
                _make_unit(
                    f"w1_{i}",
                    "Cooking recipe",
                    "ingredients flour sugar butter eggs baking cake pastry oven",
                    _BASE + timedelta(hours=i),
                )
            )
        # Window 2: all about quantum physics
        for i in range(5):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}",
                    "Quantum physics",
                    "entanglement superposition qubit quantum mechanics photon wave particle",
                    _BASE + timedelta(days=8, hours=i),
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.1)
        assert len(drifts) >= 1
        assert drifts[0]["drift_score"] > 0.5

    def test_drift_event_structure(self, store: Store):
        """Drift events have all required keys."""
        for i in range(3):
            store.insert_unit(
                _make_unit(f"w1_{i}", "Alpha", "alpha beta gamma", _BASE + timedelta(hours=i))
            )
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}",
                    "Delta",
                    "delta epsilon zeta",
                    _BASE + timedelta(days=8, hours=i),
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        assert len(drifts) >= 1
        event = drifts[0]
        assert "window_start" in event
        assert "window_end" in event
        assert "drift_score" in event
        assert "emerging_terms" in event
        assert "declining_terms" in event
        assert "top_terms_before" in event
        assert "top_terms_after" in event
        assert "unit_count_before" in event
        assert "unit_count_after" in event
        assert isinstance(event["drift_score"], float)
        assert 0.0 <= event["drift_score"] <= 1.0

    def test_emerging_and_declining_terms(self, store: Store):
        """Terms that appear/disappear between windows are correctly classified."""
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w1_{i}",
                    "Window one",
                    "alpha bravo charlie delta echo",
                    _BASE + timedelta(hours=i),
                )
            )
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}",
                    "Window two",
                    "foxtrot golf hotel india juliet",
                    _BASE + timedelta(days=8, hours=i),
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        assert len(drifts) >= 1
        event = drifts[0]

        # Terms from window 2 not in window 1
        assert "foxtrot" in event["emerging_terms"]
        # Terms from window 1 not in window 2
        assert "alpha" in event["declining_terms"]

    def test_min_drift_score_filtering(self, store: Store):
        """Events below min_drift_score are filtered out."""
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w1_{i}", "Similar", "python code programming",
                    _BASE + timedelta(hours=i)
                )
            )
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}", "Similar but different", "python code development",
                    _BASE + timedelta(days=8, hours=i)
                )
            )

        # Very low threshold: should capture drift
        drifts_low = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        # Very high threshold: likely filters it out
        drifts_high = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.99)
        assert len(drifts_low) >= len(drifts_high)

    def test_source_filter(self, store: Store):
        """Only units matching source_filter are analyzed."""
        # Window 1 from FORTY_TWO
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"ft_{i}", "Alpha", "alpha bravo charlie",
                    _BASE + timedelta(hours=i),
                    source_project=SourceProject.FORTY_TWO,
                )
            )
        # Window 2 from FORTY_TWO (different content)
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"ft2_{i}", "Delta", "delta epsilon zeta",
                    _BASE + timedelta(days=8, hours=i),
                    source_project=SourceProject.FORTY_TWO,
                )
            )
        # Window 1 from MAX (same as window 2 of FORTY_TWO, to not trigger drift)
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"max_{i}", "Max stuff", "kappa lambda mu",
                    _BASE + timedelta(hours=i),
                    source_project=SourceProject.MAX,
                )
            )

        # Without filter: should detect drift
        drifts_all = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)

        # Filter to only MAX (single window → no drift)
        drifts_max = store.detect_concept_drift(
            timedelta(days=7), min_drift_score=0.0, source_filter=SourceProject.MAX
        )
        assert drifts_max == []

        # Filter to FORTY_TWO: should detect drift
        drifts_ft = store.detect_concept_drift(
            timedelta(days=7), min_drift_score=0.0, source_filter=SourceProject.FORTY_TWO
        )
        assert len(drifts_ft) >= 1

    def test_unit_counts_per_window(self, store: Store):
        """unit_count_before and unit_count_after reflect actual counts."""
        for i in range(2):
            store.insert_unit(
                _make_unit(f"w1_{i}", "Before", "alpha beta", _BASE + timedelta(hours=i))
            )
        for i in range(5):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}", "After", "gamma delta",
                    _BASE + timedelta(days=8, hours=i)
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        assert len(drifts) >= 1
        assert drifts[0]["unit_count_before"] == 2
        assert drifts[0]["unit_count_after"] == 5

    def test_multiple_windows(self, store: Store):
        """Multiple consecutive windows produce multiple drift events."""
        topics = [
            ("cooking", "flour sugar butter eggs baking cake pastry"),
            ("physics", "quantum mechanics photon wave particle entanglement"),
            ("music", "guitar piano drums bass melody harmony rhythm"),
        ]
        for week_idx, (topic, content) in enumerate(topics):
            for i in range(3):
                store.insert_unit(
                    _make_unit(
                        f"w{week_idx}_{i}",
                        topic,
                        content,
                        _BASE + timedelta(days=7 * week_idx, hours=i),
                    )
                )

        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        # 3 windows → 2 consecutive comparisons → up to 2 drift events
        assert len(drifts) == 2

    def test_top_terms_ordering(self, store: Store):
        """top_terms_before and top_terms_after are ordered by frequency."""
        # Use repeated terms to establish clear ranking
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w1_{i}",
                    "Test",
                    "apple apple apple banana banana cherry",
                    _BASE + timedelta(hours=i),
                )
            )
        for i in range(3):
            store.insert_unit(
                _make_unit(
                    f"w2_{i}",
                    "Test",
                    "xylophone xylophone yacht zebra",
                    _BASE + timedelta(days=8, hours=i),
                )
            )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        assert len(drifts) >= 1
        # "apple" should be the most frequent term before
        before = drifts[0]["top_terms_before"]
        assert "apple" in before
        # "apple" should come before "cherry" in the list
        if "cherry" in before:
            assert before.index("apple") < before.index("cherry")

    def test_zero_time_window(self, store: Store):
        """Zero-length time window should return empty (no valid windows)."""
        store.insert_unit(
            _make_unit("n1", "Test", "content", _BASE)
        )
        drifts = store.detect_concept_drift(timedelta(seconds=0))
        assert drifts == []

    def test_window_boundaries(self, store: Store):
        """Window start/end times in results correspond to actual boundaries."""
        store.insert_unit(
            _make_unit("w1_0", "Alpha", "alpha content", _BASE)
        )
        store.insert_unit(
            _make_unit("w2_0", "Beta", "beta content", _BASE + timedelta(days=7))
        )
        drifts = store.detect_concept_drift(timedelta(days=7), min_drift_score=0.0)
        assert len(drifts) >= 1
        # window_start should be parseable as datetime
        start = datetime.fromisoformat(drifts[0]["window_start"])
        end = datetime.fromisoformat(drifts[0]["window_end"])
        assert start < end
