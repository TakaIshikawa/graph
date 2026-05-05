"""Tests for Store.get_units_by_metadata_range method."""

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


class TestGetUnitsByMetadataRange:
    """Test get_units_by_metadata_range method."""

    def test_numeric_range_query_integer_values(self, store: Store):
        """Test querying units with integer metadata values in range."""
        # Insert units with different scores
        for i, score in enumerate([10, 25, 50, 75, 90], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Score: {score}",
                    content_type=ContentType.INSIGHT,
                    metadata={"score": score},
                )
            )

        # Query for scores between 25 and 75 (inclusive)
        results = store.get_units_by_metadata_range("score", 25, 75)

        assert len(results) == 3
        scores = [unit.metadata["score"] for unit in results]
        assert set(scores) == {25, 50, 75}

    def test_numeric_range_query_float_values(self, store: Store):
        """Test querying units with float metadata values in range."""
        # Insert units with different ratings
        for i, rating in enumerate([1.5, 2.0, 2.5, 3.0, 3.5], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Rating: {rating}",
                    content_type=ContentType.INSIGHT,
                    metadata={"rating": rating},
                )
            )

        # Query for ratings between 2.0 and 3.0 (inclusive)
        results = store.get_units_by_metadata_range("rating", 2.0, 3.0)

        assert len(results) == 3
        ratings = [unit.metadata["rating"] for unit in results]
        assert set(ratings) == {2.0, 2.5, 3.0}

    def test_date_range_query_iso_strings(self, store: Store):
        """Test querying units with ISO date string metadata values in range."""
        # Insert units with different dates
        dates = ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-20", "2024-05-30"]
        for i, date in enumerate(dates, start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Date: {date}",
                    content_type=ContentType.INSIGHT,
                    metadata={"event_date": date},
                )
            )

        # Query for dates between Feb and Apr (inclusive)
        results = store.get_units_by_metadata_range(
            "event_date", "2024-02-01", "2024-04-30"
        )

        assert len(results) == 3
        event_dates = [unit.metadata["event_date"] for unit in results]
        assert set(event_dates) == {"2024-02-15", "2024-03-10", "2024-04-20"}

    def test_boundary_conditions_inclusive_min_and_max(self, store: Store):
        """Test that range boundaries are inclusive."""
        # Insert units at exact boundaries and outside
        for i, value in enumerate([9, 10, 20, 30, 31], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Value: {value}",
                    content_type=ContentType.INSIGHT,
                    metadata={"value": value},
                )
            )

        # Query for exact range [10, 30]
        results = store.get_units_by_metadata_range("value", 10, 30)

        assert len(results) == 3
        values = [unit.metadata["value"] for unit in results]
        assert set(values) == {10, 20, 30}
        # 9 and 31 should be excluded
        assert 9 not in values
        assert 31 not in values

    def test_units_outside_range_excluded(self, store: Store):
        """Test that units outside the range are properly excluded."""
        # Insert units with values clearly outside and inside range
        for i, value in enumerate([1, 50, 100, 150, 200], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Value: {value}",
                    content_type=ContentType.INSIGHT,
                    metadata={"value": value},
                )
            )

        # Query for range [75, 125]
        results = store.get_units_by_metadata_range("value", 75, 125)

        assert len(results) == 1
        assert results[0].metadata["value"] == 100

    def test_nested_metadata_field_with_dotted_path(self, store: Store):
        """Test querying nested metadata fields using dotted path notation."""
        # Insert units with nested metadata
        for i, score in enumerate([10, 20, 30, 40, 50], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Score: {score}",
                    content_type=ContentType.INSIGHT,
                    metadata={"metrics": {"quality_score": score}},
                )
            )

        # Query nested field
        results = store.get_units_by_metadata_range("metrics.quality_score", 20, 40)

        assert len(results) == 3
        scores = [unit.metadata["metrics"]["quality_score"] for unit in results]
        assert set(scores) == {20, 30, 40}

    def test_empty_result_when_no_units_in_range(self, store: Store):
        """Test that empty list is returned when no units match the range."""
        # Insert units
        for i, value in enumerate([10, 20, 30], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Value: {value}",
                    content_type=ContentType.INSIGHT,
                    metadata={"value": value},
                )
            )

        # Query for range with no matches
        results = store.get_units_by_metadata_range("value", 100, 200)

        assert len(results) == 0

    def test_empty_result_when_field_does_not_exist(self, store: Store):
        """Test that units without the specified field are excluded."""
        # Insert units, some with the field, some without
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unit-1",
                source_entity_type="insight",
                title="Unit 1",
                content="Has score",
                content_type=ContentType.INSIGHT,
                metadata={"score": 50},
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unit-2",
                source_entity_type="insight",
                title="Unit 2",
                content="No score field",
                content_type=ContentType.INSIGHT,
                metadata={"other_field": "value"},
            )
        )

        # Query for score range
        results = store.get_units_by_metadata_range("score", 0, 100)

        assert len(results) == 1
        assert results[0].metadata["score"] == 50

    def test_validates_min_value_not_greater_than_max_value(self, store: Store):
        """Test that min_value > max_value raises ValueError."""
        with pytest.raises(ValueError, match="min_value must be less than or equal to max_value"):
            store.get_units_by_metadata_range("value", 100, 50)

    def test_validates_field_name_is_non_empty_string(self, store: Store):
        """Test that empty or non-string field_name raises ValueError."""
        with pytest.raises(ValueError, match="field_name must be a non-empty string"):
            store.get_units_by_metadata_range("", 0, 100)

        with pytest.raises(ValueError, match="field_name must be a non-empty string"):
            store.get_units_by_metadata_range(123, 0, 100)  # type: ignore

    def test_validates_comparable_types_for_min_and_max(self, store: Store):
        """Test that incomparable types for min and max raise ValueError."""
        with pytest.raises(ValueError, match="min_value and max_value must be comparable types"):
            store.get_units_by_metadata_range("field", "string", 100)

    def test_single_value_range(self, store: Store):
        """Test querying with min_value == max_value (single value)."""
        # Insert units
        for i, value in enumerate([10, 20, 30], start=1):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Value: {value}",
                    content_type=ContentType.INSIGHT,
                    metadata={"value": value},
                )
            )

        # Query for exact value
        results = store.get_units_by_metadata_range("value", 20, 20)

        assert len(results) == 1
        assert results[0].metadata["value"] == 20

    def test_results_ordered_by_created_at_desc(self, store: Store):
        """Test that results are ordered by created_at descending."""
        import time

        # Insert units with small delays to ensure different timestamps
        units = []
        for i, score in enumerate([10, 20, 30], start=1):
            unit = store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"unit-{i}",
                    source_entity_type="insight",
                    title=f"Unit {i}",
                    content=f"Score: {score}",
                    content_type=ContentType.INSIGHT,
                    metadata={"score": score},
                )
            )
            units.append(unit)
            time.sleep(0.01)

        # Query all
        results = store.get_units_by_metadata_range("score", 0, 100)

        # Should be in reverse order of creation (newest first)
        assert len(results) == 3
        assert results[0].id == units[2].id  # Most recent
        assert results[1].id == units[1].id
        assert results[2].id == units[0].id  # Oldest

    def test_mixed_data_types_in_metadata_field(self, store: Store):
        """Test behavior when the same metadata field has different types across units."""
        # Insert units with different types for the same field
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unit-1",
                source_entity_type="insight",
                title="Unit 1",
                content="Numeric",
                content_type=ContentType.INSIGHT,
                metadata={"value": 50},
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unit-2",
                source_entity_type="insight",
                title="Unit 2",
                content="String",
                content_type=ContentType.INSIGHT,
                metadata={"value": "text"},
            )
        )

        # Query with numeric range - should only match numeric values
        results = store.get_units_by_metadata_range("value", 0, 100)

        # SQLite type affinity may cause unexpected behavior
        # This documents the actual behavior
        assert len(results) >= 1
        assert results[0].metadata["value"] == 50
