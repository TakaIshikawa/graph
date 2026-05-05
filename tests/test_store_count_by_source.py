"""Tests for Store.count_units_by_source method."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

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


class TestCountUnitsBySource:
    """Test count_units_by_source method."""

    def test_basic_counting_across_multiple_source_types(self, store: Store):
        """Test basic unit counting grouped by source project."""
        # Insert units from different source projects
        for _ in range(3):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"max-{_}",
                    source_entity_type="insight",
                    title=f"MAX Unit {_}",
                    content="From MAX",
                    content_type=ContentType.INSIGHT,
                )
            )

        for _ in range(2):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.FORTY_TWO,
                    source_id=f"42-{_}",
                    source_entity_type="insight",
                    title=f"42 Unit {_}",
                    content="From FORTY_TWO",
                    content_type=ContentType.INSIGHT,
                )
            )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="presence-1",
                source_entity_type="insight",
                title="Presence Unit",
                content="From PRESENCE",
                content_type=ContentType.INSIGHT,
            )
        )

        result = store.count_units_by_source()

        assert len(result) == 3
        assert result[SourceProject.MAX] == 3
        assert result[SourceProject.FORTY_TWO] == 2
        assert result[SourceProject.PRESENCE] == 1

    def test_time_range_filtering(self, store: Store):
        """Test filtering by time range."""
        # Insert units at different times
        old_unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="old",
                source_entity_type="insight",
                title="Old Unit",
                content="Old",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        )

        recent_unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="recent",
                source_entity_type="insight",
                title="Recent Unit",
                content="Recent",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        )

        # Filter for recent units only
        time_range = (
            datetime(2024, 5, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        result = store.count_units_by_source(time_range=time_range)

        # Should only count the recent unit
        assert result[SourceProject.MAX] == 1

    def test_tag_filtering(self, store: Store):
        """Test filtering by tags."""
        # Insert units with different tags
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-1",
                source_entity_type="insight",
                title="Solar Unit 1",
                content="Solar energy",
                content_type=ContentType.INSIGHT,
                tags=["solar", "energy"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-2",
                source_entity_type="insight",
                title="Solar Unit 2",
                content="Solar panels",
                content_type=ContentType.INSIGHT,
                tags=["solar", "panels"],
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="wind-1",
                source_entity_type="insight",
                title="Wind Unit",
                content="Wind energy",
                content_type=ContentType.INSIGHT,
                tags=["wind", "energy"],
            )
        )

        # Filter for units with "solar" tag
        result = store.count_units_by_source(tags=["solar"])

        assert result[SourceProject.MAX] == 2
        assert SourceProject.FORTY_TWO not in result

    def test_tag_filtering_requires_all_tags(self, store: Store):
        """Test that tag filtering requires ALL specified tags."""
        # Insert units with different tag combinations
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="both-tags",
                source_entity_type="insight",
                title="Both Tags",
                content="Has both",
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
                content="Only solar",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        # Filter for units with BOTH "solar" AND "energy"
        result = store.count_units_by_source(tags=["solar", "energy"])

        # Only the first unit should match
        assert result[SourceProject.MAX] == 1

    def test_metadata_key_filtering(self, store: Store):
        """Test filtering by metadata key presence."""
        # Insert units with and without specific metadata
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="with-score",
                source_entity_type="insight",
                title="With Score",
                content="Has score",
                content_type=ContentType.INSIGHT,
                metadata={"score": 95},
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="without-score",
                source_entity_type="insight",
                title="Without Score",
                content="No score",
                content_type=ContentType.INSIGHT,
                metadata={"other": "value"},
            )
        )

        # Filter for units with "score" metadata
        result = store.count_units_by_source(metadata_key="score")

        assert result[SourceProject.MAX] == 1

    def test_combined_filters(self, store: Store):
        """Test combining multiple filters together."""
        # Insert units with various combinations
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="matches-all",
                source_entity_type="insight",
                title="Matches All",
                content="Perfect match",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                tags=["solar"],
                metadata={"score": 95},
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="wrong-time",
                source_entity_type="insight",
                title="Wrong Time",
                content="Old",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                tags=["solar"],
                metadata={"score": 80},
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="wrong-tag",
                source_entity_type="insight",
                title="Wrong Tag",
                content="Different tag",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                tags=["wind"],
                metadata={"score": 90},
            )
        )

        # Apply all filters
        time_range = (
            datetime(2024, 5, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        result = store.count_units_by_source(
            time_range=time_range,
            tags=["solar"],
            metadata_key="score",
        )

        # Only the first unit matches all criteria
        assert result[SourceProject.MAX] == 1

    def test_empty_result_when_no_matches(self, store: Store):
        """Test that empty dict is returned when no units match filters."""
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unit-1",
                source_entity_type="insight",
                title="Unit 1",
                content="Content",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )

        # Filter for non-existent tags
        result = store.count_units_by_source(tags=["nonexistent"])

        assert len(result) == 0

    def test_empty_result_when_store_is_empty(self, store: Store):
        """Test that empty dict is returned when store has no units."""
        result = store.count_units_by_source()

        assert len(result) == 0

    def test_validates_time_range_format(self, store: Store):
        """Test that invalid time_range raises ValueError."""
        # Not a tuple
        with pytest.raises(ValueError, match="time_range must be a tuple of two datetime objects"):
            store.count_units_by_source(time_range="invalid")  # type: ignore

        # Wrong number of elements
        with pytest.raises(ValueError, match="time_range must be a tuple of two datetime objects"):
            store.count_units_by_source(
                time_range=(datetime(2024, 1, 1, tzinfo=timezone.utc),)  # type: ignore
            )

        # Not datetime objects
        with pytest.raises(ValueError, match="time_range must be a tuple of two datetime objects"):
            store.count_units_by_source(time_range=("2024-01-01", "2024-12-31"))  # type: ignore

    def test_validates_time_range_order(self, store: Store):
        """Test that time_range with start > end raises ValueError."""
        with pytest.raises(ValueError, match="time_range start must be before or equal to end"):
            store.count_units_by_source(
                time_range=(
                    datetime(2024, 12, 31, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            )

    def test_validates_tags_format(self, store: Store):
        """Test that invalid tags raises ValueError."""
        # Not a list
        with pytest.raises(ValueError, match="tags must be a list of strings"):
            store.count_units_by_source(tags="solar")  # type: ignore

        # Contains non-strings
        with pytest.raises(ValueError, match="tags must be a list of strings"):
            store.count_units_by_source(tags=["solar", 123])  # type: ignore

    def test_validates_metadata_key_format(self, store: Store):
        """Test that invalid metadata_key raises ValueError."""
        with pytest.raises(ValueError, match="metadata_key must be a string"):
            store.count_units_by_source(metadata_key=123)  # type: ignore

    def test_nested_metadata_key_filtering(self, store: Store):
        """Test filtering by nested metadata key using dotted path."""
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="nested",
                source_entity_type="insight",
                title="Nested Metadata",
                content="Has nested",
                content_type=ContentType.INSIGHT,
                metadata={"metrics": {"quality": 95}},
            )
        )

        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="flat",
                source_entity_type="insight",
                title="Flat Metadata",
                content="No nested",
                content_type=ContentType.INSIGHT,
                metadata={"quality": 80},
            )
        )

        # Filter for nested key
        result = store.count_units_by_source(metadata_key="metrics.quality")

        assert result[SourceProject.MAX] == 1

    def test_counts_across_different_source_projects_with_filters(self, store: Store):
        """Test counting different source projects with filters applied."""
        # Insert units from multiple sources with matching filters
        for i in range(2):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"max-{i}",
                    source_entity_type="insight",
                    title=f"MAX {i}",
                    content="Content",
                    content_type=ContentType.INSIGHT,
                    tags=["solar"],
                )
            )

        for i in range(3):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.FORTY_TWO,
                    source_id=f"42-{i}",
                    source_entity_type="insight",
                    title=f"42 {i}",
                    content="Content",
                    content_type=ContentType.INSIGHT,
                    tags=["solar"],
                )
            )

        result = store.count_units_by_source(tags=["solar"])

        assert result[SourceProject.MAX] == 2
        assert result[SourceProject.FORTY_TWO] == 3
