from __future__ import annotations

from graph.adapters.goodreads_books_csv import GoodreadsBooksCsvAdapter


def test_goodreads_books_csv_parses_headers_ratings_and_shelves(tmp_path):
    path = tmp_path / "goodreads.csv"
    path.write_text(
        "Book Id,Title,Author,ISBN,My Rating,Average Rating,Bookshelves,Date Read,Date Added,My Review\n"
        "10,The Book,Ann Author,123,5,4.2,\"read, favorites\",2026-01-02,2026-01-01,Loved it\n"
        "11,No Review,Bob Writer,,3,3.5,to-read,,2026-01-03,\n",
        encoding="utf-8",
    )

    units = GoodreadsBooksCsvAdapter(path=str(path)).ingest().units

    assert [unit.title for unit in units] == ["The Book", "No Review"]
    assert units[0].metadata["author"] == "Ann Author"
    assert units[0].metadata["isbn"] == "123"
    assert units[0].metadata["my_rating"] == "5"
    assert units[0].metadata["average_rating"] == "4.2"
    assert units[0].metadata["bookshelves"] == ["read", "favorites"]
    assert "Loved it" in units[0].content
    assert "Author: Bob Writer" in units[1].content
