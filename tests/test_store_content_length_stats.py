"""Tests for Store.get_content_length_distribution_stats method."""

from __future__ import annotations

import os
import tempfile

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


class TestGetContentLengthDistributionStats:
    """Test get_content_length_distribution_stats method."""

    def test_basic_statistics_calculation(self, store: Store):
        """Test basic statistics with simple data."""
        # Insert units with known content lengths
        # Lengths: 10, 20, 30, 40, 50
        for i in range(1, 6):
            content = "x" * (i * 10)
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=content,
                    content_type=ContentType.INSIGHT,
                )
            )

        stats = store.get_content_length_distribution_stats()

        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["p50"] == 30.0

    def test_percentile_calculations(self, store: Store):
        """Test percentile calculations with larger dataset."""
        # Insert 100 units with lengths 1-100
        for i in range(1, 101):
            content = "x" * i
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=content,
                    content_type=ContentType.INSIGHT,
                )
            )

        stats = store.get_content_length_distribution_stats()

        # For uniform distribution 1-100
        assert 49.0 <= stats["p50"] <= 51.0  # Should be around 50
        assert 74.0 <= stats["p75"] <= 76.0  # Should be around 75
        assert 89.0 <= stats["p90"] <= 91.0  # Should be around 90
        assert 94.0 <= stats["p95"] <= 96.0  # Should be around 95
        assert 98.0 <= stats["p99"] <= 100.0  # Should be around 99
        assert stats["max"] == 100.0
        assert 50.0 <= stats["mean"] <= 51.0  # Average of 1-100 is 50.5

    def test_filtering_by_source_project(self, store: Store):
        """Test statistics filtered by source project."""
        # Insert units from different projects with different length patterns
        # MAX: lengths 10, 20, 30
        for i in range(1, 4):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"max-{i}",
                    source_entity_type="insight",
                    title=f"MAX {i}",
                    content="x" * (i * 10),
                    content_type=ContentType.INSIGHT,
                )
            )

        # FORTY_TWO: lengths 100, 200
        for i in range(1, 3):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.FORTY_TWO,
                    source_id=f"42-{i}",
                    source_entity_type="insight",
                    title=f"42 {i}",
                    content="x" * (i * 100),
                    content_type=ContentType.INSIGHT,
                )
            )

        # Get stats for MAX only
        max_stats = store.get_content_length_distribution_stats(source_project=SourceProject.MAX)

        assert max_stats["max"] == 30.0
        assert max_stats["mean"] == 20.0
        assert max_stats["median"] == 20.0

    def test_filtering_by_tags(self, store: Store):
        """Test statistics filtered by tags."""
        # Insert units with different tags and lengths
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-1",
                source_entity_type="insight",
                title="Solar 1",
                content="x" * 10,
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-2",
                source_entity_type="insight",
                title="Solar 2",
                content="x" * 20,
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="wind-1",
                source_entity_type="insight",
                title="Wind 1",
                content="x" * 1000,
                content_type=ContentType.INSIGHT,
                tags=["wind"],
            )
        )

        # Get stats for solar tags only
        solar_stats = store.get_content_length_distribution_stats(tags=["solar"])

        assert solar_stats["max"] == 20.0
        assert solar_stats["mean"] == 15.0

    def test_combined_filtering(self, store: Store):
        """Test statistics with both source project and tag filters."""
        # Insert units with various combinations
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="match",
                source_entity_type="insight",
                title="Match",
                content="x" * 50,
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="wrong-tag",
                source_entity_type="insight",
                title="Wrong Tag",
                content="x" * 1000,
                content_type=ContentType.INSIGHT,
                tags=["wind"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="wrong-project",
                source_entity_type="insight",
                title="Wrong Project",
                content="x" * 2000,
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        # Filter for MAX + solar
        stats = store.get_content_length_distribution_stats(
            source_project=SourceProject.MAX,
            tags=["solar"],
        )

        # Should only include the first unit (length 50)
        assert stats["max"] == 50.0
        assert stats["mean"] == 50.0
        assert stats["median"] == 50.0

    def test_empty_result_handling(self, store: Store):
        """Test that zeros are returned when no units match filters."""
        stats = store.get_content_length_distribution_stats()

        assert stats["p50"] == 0.0
        assert stats["p75"] == 0.0
        assert stats["p90"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["p99"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["max"] == 0.0

    def test_single_unit_statistics(self, store: Store):
        """Test statistics when only one unit exists."""
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="only",
                source_entity_type="insight",
                title="Only Unit",
                content="x" * 42,
                content_type=ContentType.INSIGHT,
            )
        )

        stats = store.get_content_length_distribution_stats()

        # All percentiles should equal the single value
        assert stats["p50"] == 42.0
        assert stats["p75"] == 42.0
        assert stats["p90"] == 42.0
        assert stats["p95"] == 42.0
        assert stats["p99"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["median"] == 42.0
        assert stats["max"] == 42.0

    def test_validates_source_project_type(self, store: Store):
        """Test that non-SourceProject values raise TypeError."""
        with pytest.raises(TypeError, match="source_project must be a SourceProject enum"):
            store.get_content_length_distribution_stats(source_project="max")  # type: ignore

    def test_validates_tags_format(self, store: Store):
        """Test that invalid tags raise ValueError."""
        with pytest.raises(ValueError, match="tags must be a list of strings"):
            store.get_content_length_distribution_stats(tags="solar")  # type: ignore

        with pytest.raises(ValueError, match="tags must be a list of strings"):
            store.get_content_length_distribution_stats(tags=["solar", 123])  # type: ignore

    def test_skewed_distribution(self, store: Store):
        """Test statistics with skewed distribution."""
        # Insert many small units and a few large ones
        for i in range(95):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"small-{i}",
                    source_entity_type="insight",
                    title=f"Small {i}",
                    content="x" * 10,
                    content_type=ContentType.INSIGHT,
                )
            )

        for i in range(5):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"large-{i}",
                    source_entity_type="insight",
                    title=f"Large {i}",
                    content="x" * 1000,
                    content_type=ContentType.INSIGHT,
                )
            )

        stats = store.get_content_length_distribution_stats()

        # Most percentiles should be near 10, but p95, p99 and max should be much higher
        assert stats["p50"] == 10.0
        assert stats["p75"] == 10.0
        assert stats["p90"] == 10.0
        # p95 is at the 95th position out of 100, which falls in the transition zone
        assert stats["p95"] > 10.0  # Should be higher than median
        assert stats["p99"] > 100.0  # Should be much higher
        assert stats["max"] == 1000.0
        # Mean should be higher than median due to skew
        assert stats["mean"] > stats["median"]

    def test_empty_content_units(self, store: Store):
        """Test statistics with units that have empty content."""
        # Insert units with varying content lengths including empty
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="empty",
                source_entity_type="insight",
                title="Empty",
                content="",
                content_type=ContentType.INSIGHT,
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="small",
                source_entity_type="insight",
                title="Small",
                content="x" * 10,
                content_type=ContentType.INSIGHT,
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="large",
                source_entity_type="insight",
                title="Large",
                content="x" * 20,
                content_type=ContentType.INSIGHT,
            )
        )

        stats = store.get_content_length_distribution_stats()

        # Should include the empty content (length 0)
        assert stats["median"] == 10.0
        assert stats["mean"] == 10.0  # (0 + 10 + 20) / 3
        assert stats["max"] == 20.0

    def test_tag_filtering_requires_all_tags(self, store: Store):
        """Test that tag filtering requires ALL specified tags."""
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="both",
                source_entity_type="insight",
                title="Both Tags",
                content="x" * 100,
                content_type=ContentType.INSIGHT,
                tags=["solar", "energy"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-only",
                source_entity_type="insight",
                title="Solar Only",
                content="x" * 200,
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        # Filter for both tags
        stats = store.get_content_length_distribution_stats(tags=["solar", "energy"])

        # Should only include first unit
        assert stats["max"] == 100.0
        assert stats["mean"] == 100.0
