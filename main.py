"""
NASCAR Historical Stats Builder
--------------------------------
Pulls NASCAR Cup Series standings from ESPN's free public API
and stores them in a local SQLite database (nascar.db).

No API key required. No account required.

Run this script with:   python main.py
Query your data with:   python query.py

Data available: ~78 seasons (back to the late 1940s)
We default to pulling 2000-2024 for clean, modern data.
"""

import requests
import sqlite3
import time
import re

# ── Configuration ─────────────────────────────────────────────────────────────
DB_FILE    = "nascar.db"
START_YEAR = 2000
END_YEAR   = 2024
PAUSE      = 0.3   # seconds between API calls - be polite to ESPN's servers

# ── ESPN API helpers ───────────────────────────────────────────────────────────
BASE = "http://sports.core.api.espn.com/v2/sports/racing/leagues/nascar-premier"

def get(url):
    """Make a GET request and return JSON. Retries once on failure."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [retry] {e}")
        time.sleep(2)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e2:
            print(f"    [failed] {e2}")
            return None

def get_standings(year):
    """Fetch the full standings list for a given year."""
    url = f"{BASE}/seasons/{year}/types/2/standings/0?lang=en&region=us&limit=100"
    return get(url)

def get_driver_name(athlete_id, year):
    """
    Fetch a driver's name from ESPN using the season-level athletes endpoint.
    Returns (driver_id, first_name, last_name, display_name) or None.
    """
    url = f"{BASE}/seasons/{year}/athletes/{athlete_id}?lang=en&region=us"
    data = get(url)
    if data is None or "error" in data:
        return None
    return (
        int(data.get("id", 0)),
        data.get("firstName", ""),
        data.get("lastName", ""),
        data.get("displayName", ""),
    )

def extract_athlete_id(ref_url):
    """Pull the numeric athlete ID out of an ESPN ref URL."""
    match = re.search(r"/athletes/(\d+)/", ref_url)
    return int(match.group(1)) if match else None

def extract_stat(stats_list, name):
    """Find a stat by name in a list of stat objects."""
    for s in stats_list:
        if s.get("name") == name:
            return s.get("value")
    return None

# ── Database setup ─────────────────────────────────────────────────────────────
def setup_database(conn):
    """Create tables if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS drivers (
            id           INTEGER PRIMARY KEY,
            first_name   TEXT,
            last_name    TEXT,
            display_name TEXT
        );

        CREATE TABLE IF NOT EXISTS season_standings (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id         INTEGER NOT NULL,
            year              INTEGER NOT NULL,
            rank              INTEGER,
            wins              INTEGER,
            top5              INTEGER,
            top10             INTEGER,
            poles             INTEGER,
            starts            INTEGER,
            dnf               INTEGER,
            laps_led          INTEGER,
            championship_pts  INTEGER,
            bonus_pts         INTEGER,
            penalty_pts       INTEGER,
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (driver_id, year)
        );
    """)
    conn.commit()
    print("Database ready.")

def upsert_driver(conn, driver_id, first_name, last_name, display_name):
    """Insert a driver if they don't exist yet."""
    conn.execute("""
        INSERT OR IGNORE INTO drivers (id, first_name, last_name, display_name)
        VALUES (?, ?, ?, ?)
    """, (driver_id, first_name, last_name, display_name))

def upsert_standing(conn, driver_id, year, stats):
    """Insert or replace a driver's season standing."""
    conn.execute("""
        INSERT OR REPLACE INTO season_standings
            (driver_id, year, rank, wins, top5, top10, poles, starts,
             dnf, laps_led, championship_pts, bonus_pts, penalty_pts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        driver_id, year,
        stats.get("rank"),
        stats.get("wins"),
        stats.get("top5"),
        stats.get("top10"),
        stats.get("poles"),
        stats.get("starts"),
        stats.get("dnf"),
        stats.get("laps_led"),
        stats.get("championship_pts"),
        stats.get("bonus_pts"),
        stats.get("penalty_pts"),
    ))

# ── Main fetch loop ────────────────────────────────────────────────────────────
def fetch_year(conn, year, known_drivers):
    """Fetch and store all driver standings for one season."""
    print(f"\n-- {year} ---------------------")
    data = get_standings(year)
    if not data or "standings" not in data:
        print(f"  No standings data found for {year}.")
        return

    entries = data["standings"]
    print(f"  Found {len(entries)} driver entries.")

    for entry in entries:
        records = entry.get("records", [])
        if not records:
            continue

        record      = records[0]
        athlete_ref = record.get("$ref", "")
        stats_list  = record.get("stats", [])

        if not athlete_ref:
            continue

        # Extract athlete ID from the ref URL
        # e.g. ".../athletes/4319/records/0"
        athlete_id = extract_athlete_id(athlete_ref)
        if not athlete_id:
            continue

        # Fetch driver name only if we haven't seen this driver yet
        if athlete_id not in known_drivers:
            time.sleep(PAUSE)
            info = get_driver_name(athlete_id, year)
            if info:
                driver_id, first, last, display = info
                upsert_driver(conn, driver_id, first, last, display)
                known_drivers.add(driver_id)
                print(f"    + Added driver: {display}")
            else:
                continue
        else:
            driver_id = athlete_id

        # Parse stats
        stats = {
            "rank":            int(extract_stat(stats_list, "rank")   or 0),
            "wins":            int(extract_stat(stats_list, "wins")   or 0),
            "top5":            int(extract_stat(stats_list, "top5")   or 0),
            "top10":           int(extract_stat(stats_list, "top10")  or 0),
            "poles":           int(extract_stat(stats_list, "poles")  or 0),
            "starts":          int(extract_stat(stats_list, "starts") or 0),
            "dnf":             int(extract_stat(stats_list, "dnf")    or 0),
            "laps_led":        int(extract_stat(stats_list, "lapsLead")        or 0),
            "championship_pts":int(extract_stat(stats_list, "championshipPts") or 0),
            "bonus_pts":       int(extract_stat(stats_list, "bonus")           or 0),
            "penalty_pts":     int(extract_stat(stats_list, "penaltyPts")      or 0),
        }

        upsert_standing(conn, driver_id, year, stats)

    conn.commit()
    print(f"  Saved {year} standings.")
    time.sleep(PAUSE)

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  NASCAR Historical Stats Builder")
    print(f"  Pulling seasons {START_YEAR} - {END_YEAR}")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)
    setup_database(conn)

    known_drivers = set()

    for year in range(START_YEAR, END_YEAR + 1):
        fetch_year(conn, year, known_drivers)

    conn.close()

    print("\n" + "=" * 50)
    print(f"  Done! Database saved to: {DB_FILE}")
    print("  Run 'python query.py' to explore your data.")
    print("=" * 50)

if __name__ == "__main__":
    main()
