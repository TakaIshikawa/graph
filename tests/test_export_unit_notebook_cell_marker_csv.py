import csv
from io import StringIO

from graph.export import export_units_to_notebook_cell_marker_csv


def test_notebook_cell_marker_export_detects_python_and_markdown_markers():
    content = "# %%\nprint(1)\n<!-- %% [markdown] -->\nordinary # %% text\n```\nnot marker\n```"
    rows = list(csv.DictReader(StringIO(export_units_to_notebook_cell_marker_csv([{"id": "u1", "content": content}]))))

    assert rows == [
        {"unit_id": "u1", "title": "", "line_number": "1", "marker_type": "percent", "language": "", "raw_marker": "# %%"},
        {"unit_id": "u1", "title": "", "line_number": "3", "marker_type": "%%", "language": "markdown", "raw_marker": "<!-- %% [markdown] -->"},
    ]
