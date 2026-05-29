from graph.adapters import SteamPlaytimeCsvAdapter


def test_steam_playtime_csv_normalizes_minutes_and_hours(tmp_path):
    path = tmp_path / "steam.csv"
    path.write_text("app id,game name,playtime minutes,playtime hours,last played,platform,store URL\n10,Game,90,,2025-01-02,pc,https://store.test/10\n20,Other,,,2025-01-03,deck,\n30,Hours,,2.5,,pc,\n", encoding="utf-8")

    units = SteamPlaytimeCsvAdapter(str(path)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert by_title["Game"].metadata["playtime_minutes"] == 90
    assert by_title["Other"].metadata.get("url") is None
    assert by_title["Hours"].metadata["playtime_minutes"] == 150
