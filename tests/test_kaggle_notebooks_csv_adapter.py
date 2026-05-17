from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.kaggle_notebooks_csv import KaggleNotebooksCsvAdapter
from graph.types.enums import EdgeRelation
from graph.types.models import SyncState


def test_kaggle_notebooks_csv_ingests_normalized_notebooks(tmp_path):
    export = tmp_path / "notebooks.csv"
    export.write_text(
        "Notebook Id,Title,Notebook URL,Author,Dataset,Competition,Language,Votes,Views,Comments,Last Run Time,Created Date,Tags,Description\n"
        "nb-1,Titanic EDA,https://www.kaggle.com/code/ada/titanic-eda,Ada,Titanic,Titanic ML,Python,42,1200,7,2026-05-02 10:30:00,2026-05-01,\"eda, tutorial\",Feature exploration\n",
        encoding="utf-8",
    )

    result = KaggleNotebooksCsvAdapter(path=str(export)).ingest(entity_types=["notebook"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "kaggle_notebooks_csv"
    assert unit.source_entity_type == "notebook"
    assert unit.source_id == KaggleNotebooksCsvAdapter(path=str(export)).ingest(entity_types=["notebook"]).units[0].source_id
    assert unit.metadata["notebook_id"] == "nb-1"
    assert unit.metadata["title"] == "Titanic EDA"
    assert unit.metadata["notebook_url"] == "https://www.kaggle.com/code/ada/titanic-eda"
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["dataset"] == "Titanic"
    assert unit.metadata["competition"] == "Titanic ML"
    assert unit.metadata["language"] == "Python"
    assert unit.metadata["votes"] == 42
    assert unit.metadata["views"] == 1200
    assert unit.metadata["comments"] == 7
    assert unit.metadata["last_run_at"] == "2026-05-02T10:30:00+00:00"
    assert unit.metadata["created_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["tags"] == ["eda", "tutorial"]
    assert unit.metadata["description"] == "Feature exploration"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 10, 30, tzinfo=timezone.utc)


def test_kaggle_notebooks_csv_supports_directory_sparse_rows_and_fallback_ids(tmp_path):
    first = tmp_path / "first.csv"
    nested = tmp_path / "nested"
    nested.mkdir()
    first.write_text("Title,Author,Slug\nSparse Notebook,Ada,sparse-notebook\n,,\n", encoding="utf-8")
    (nested / "second.csv").write_text("Title,Notebook URL,Dataset\nURL Notebook,https://www.kaggle.com/code/bob/url,House Prices\n", encoding="utf-8")

    result = KaggleNotebooksCsvAdapter(path=str(tmp_path)).ingest(entity_types=["notebook"])
    units = sorted(result.units, key=lambda unit: unit.title)

    assert [unit.title for unit in units] == ["Sparse Notebook", "URL Notebook"]
    assert units[0].metadata["slug"] == "sparse-notebook"
    assert units[1].metadata["dataset"] == "House Prices"
    again = sorted(KaggleNotebooksCsvAdapter(path=str(tmp_path)).ingest(entity_types=["notebook"]).units, key=lambda unit: unit.title)
    assert [unit.source_id for unit in units] == [unit.source_id for unit in again]


def test_kaggle_notebooks_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "notebooks.csv"
    export.write_text(
        "Notebook Id,Title,Author,Dataset,Competition,Last Run Time,Created Date\n"
        "old,Old Notebook,Ada,Old Data,Old Comp,2026-04-01,2026-03-01\n"
        "new,New Notebook,Bob,New Data,New Comp,2026-05-03,2026-05-01\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="kaggle_notebooks_csv",
        source_entity_type="notebook",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = KaggleNotebooksCsvAdapter(path=str(export)).ingest(
        since=since,
        entity_types=["notebook", "dataset", "author", "competition"],
    )

    assert {unit.title for unit in result.units if unit.source_entity_type == "notebook"} == {"New Notebook"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "dataset"} == {"New Data"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "author"} == {"Bob"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "competition"} == {"New Comp"}
    assert KaggleNotebooksCsvAdapter(path=str(export)).ingest(entity_types=["discussion"]).units == []


def test_kaggle_notebooks_csv_emits_aggregate_units_and_edges(tmp_path):
    export = tmp_path / "notebooks.csv"
    export.write_text(
        "Notebook Id,Title,Author,Dataset,Competition,Language,Votes,Views,Comments,Last Run Time,Tags\n"
        "one,First,Ada,Titanic,Titanic ML,Python,10,100,1,2026-05-01,eda\n"
        "two,Second,Ada,Titanic,Titanic ML,R,20,200,2,2026-05-02,viz\n"
        "three,Third,Bob,House Prices,House Prices,Python,5,50,0,2026-05-03,baseline\n",
        encoding="utf-8",
    )

    result = KaggleNotebooksCsvAdapter(path=str(export)).ingest()
    datasets = {unit.title: unit for unit in result.units if unit.source_entity_type == "dataset"}
    authors = {unit.title: unit for unit in result.units if unit.source_entity_type == "author"}
    competitions = {unit.title: unit for unit in result.units if unit.source_entity_type == "competition"}

    assert datasets["Titanic"].metadata["notebook_count"] == 2
    assert datasets["Titanic"].metadata["total_votes"] == 30
    assert datasets["Titanic"].metadata["total_views"] == 300
    assert datasets["Titanic"].metadata["languages"] == ["Python", "R"]
    assert authors["Ada"].metadata["datasets"] == ["Titanic"]
    assert competitions["Titanic ML"].metadata["total_comments"] == 3
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]) == 3
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.RELATES_TO]) == 6

    author_only = KaggleNotebooksCsvAdapter(path=str(export)).ingest(entity_types=["author"])
    assert {unit.source_entity_type for unit in author_only.units} == {"author"}
    assert author_only.edges == []


def test_kaggle_notebooks_csv_handles_empty_missing_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_bytes(b"\xff\xfe\x00")

    assert KaggleNotebooksCsvAdapter(path=str(empty)).ingest().units == []
    assert KaggleNotebooksCsvAdapter(path=str(tmp_path / "missing.csv")).ingest().units == []
    assert KaggleNotebooksCsvAdapter(path=str(malformed)).ingest().units == []
