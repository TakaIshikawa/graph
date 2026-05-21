from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_geospatial_cluster_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_geospatial_cluster_csv_clusters_valid_coordinates():
    text = export_unit_geospatial_cluster_csv(
        [
            {"id": "a", "source_project": "geo", "metadata": {"lat": "35.01", "lng": "139.04"}},
            {"id": "b", "source_project": "geo", "latitude": 35.02, "longitude": 139.03},
            {"id": "bad", "source_project": "geo", "metadata": {"lat": "x", "lng": "139"}},
        ],
        precision=1,
    )

    row = rows(text)[0]
    assert row["cluster_key"] == "35.0,139.0"
    assert row["unit_count"] == "2"
    assert row["representative_unit_ids"] == "a; b"
    assert row["centroid_latitude"] == "35.015000"


def test_export_unit_geospatial_cluster_csv_path_mode(tmp_path):
    path = tmp_path / "clusters.csv"
    stats = export_unit_geospatial_cluster_csv([{"id": "a", "metadata": {"lat": 1, "lon": 2}}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["source_project"] == "Unknown"
    assert stats["rows_exported"] == 1

