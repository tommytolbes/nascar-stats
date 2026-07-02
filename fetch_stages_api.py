"""
NASCAR Stage Results Fetcher (API)
------------------------------------
Pulls FULL-FIELD stage finishing positions from NASCAR's public cacher CDN
and loads them into the stage_results table. Replaces the Jayski PDF flow
(process_stages.py), which only captured the top 10 per stage.

Data source (no API key required):
  https://cf.nascar.com/cacher/{year}/race_list_basic.json
      -> series_1 race list with NASCAR race_ids and dates
  https://cf.nascar.com/cacher/{year}/1/{race_id}/current-results.json
      -> Results[] with S1Fin / S2Fin for every driver in the field

NASCAR race_ids differ from our ESPN race ids, so races are matched by date.
NASCAR driver ids differ from our ESPN driver ids, so drivers are matched by
normalized display name (same approach as salary loading in build_fantasy.py).

Usage:   python fetch_stages_api.py [year ...]
         (defaults to the current calendar year)
"""

import datetime
import sqlite3
import sys
import time
import unicodedata

import requests

DB_FILE = "nascar.db"
PAUSE   = 0.3   # seconds between API calls

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Referer": "https://www.nascar.com/",
}

# NASCAR feed name -> drivers.display_name where normalization alone isn't enough
NAME_ALIASES = {
    "john h nemechek": "john hunter nemechek",
}


def normalize(name):
    """Lowercase, strip accents and periods so feed names match display_name."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().replace(".", "").strip()
    name = " ".join(name.split())
    return NAME_ALIASES.get(name, name)


def get_json(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 404):
                return None  # race archive not available; don't retry
        except Exception as e:
            if attempt == 2:
                print(f"    [failed] {url}: {e}")
                return None
        time.sleep(2 ** attempt)
    return None


def load_year(conn, year, name_map):
    print(f"\n-- {year} ---------------------")

    race_list = get_json(f"https://cf.nascar.com/cacher/{year}/race_list_basic.json")
    if not race_list:
        print("  Could not fetch race list — skipping year.")
        return 0

    nascar_by_date = {}
    for race in race_list.get("series_1", []):
        date = (race.get("race_date") or "")[:10]
        if date:
            nascar_by_date[date] = race

    # Our races for this year that are complete (have results)
    db_races = conn.execute("""
        SELECT r.id, r.date, r.name FROM races r
        WHERE r.year = ?
          AND EXISTS (SELECT 1 FROM race_results rr
                      WHERE rr.race_id = r.id AND rr.finish_pos IS NOT NULL)
        ORDER BY r.date
    """, (year,)).fetchall()

    loaded = 0
    for espn_race_id, date, race_name in db_races:
        nascar_race = nascar_by_date.get(date)
        if not nascar_race:
            continue  # e.g. exhibition events not in the points race list
        nascar_race_id = nascar_race["race_id"]

        # Skip races that already have full-field stage data (>10 rows means
        # we already loaded from the API, not just a top-10 PDF)
        existing = conn.execute(
            "SELECT COUNT(*) FROM stage_results WHERE race_id = ?", (espn_race_id,)
        ).fetchone()[0]
        if existing > 10:
            continue

        time.sleep(PAUSE)
        data = get_json(
            f"https://cf.nascar.com/cacher/{year}/1/{nascar_race_id}/current-results.json"
        )
        if not data or not data.get("Results"):
            print(f"  {date} {race_name[:45]}: no archive data")
            continue

        inserted = 0
        unmatched = []
        for res in data["Results"]:
            s1 = res.get("S1Fin") or None
            s2 = res.get("S2Fin") or None
            if s1 is None and s2 is None:
                continue
            driver_id = name_map.get(normalize(res.get("DriverNameTag", "")))
            if driver_id is None:
                unmatched.append(res.get("DriverNameTag", "?"))
                continue
            conn.execute("""
                INSERT OR REPLACE INTO stage_results
                    (race_id, driver_id, stage1_pos, stage2_pos)
                VALUES (?, ?, ?, ?)
            """, (espn_race_id, driver_id, s1, s2))
            inserted += 1

        conn.commit()
        loaded += 1
        note = f" (unmatched: {', '.join(unmatched)})" if unmatched else ""
        print(f"  {date} {race_name[:45]}: {inserted} drivers{note}")

    if loaded == 0:
        print("  Nothing new to load.")
    return loaded


def main():
    years = [int(a) for a in sys.argv[1:]] or [datetime.date.today().year]

    print("=" * 50)
    print("  NASCAR Stage Results Fetcher (API)")
    print(f"  Years: {years}")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)

    name_map = {
        normalize(row[1]): row[0]
        for row in conn.execute("SELECT id, display_name FROM drivers")
        if row[1]
    }

    total = 0
    for year in years:
        total += load_year(conn, year, name_map)

    conn.close()
    print(f"\nDone. Loaded stage data for {total} race(s).")
    print("Run 'python build_fantasy.py' to apply stage points to scores.")


if __name__ == "__main__":
    main()
