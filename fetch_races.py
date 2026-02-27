"""
NASCAR Race-by-Race Results Fetcher
-------------------------------------
Adds three new tables to nascar.db:
  - tracks       : track/venue details (name, city, state, length)
  - races        : one row per race (name, date, track, season)
  - race_results : one row per driver per race (finish pos, laps led, pts, etc.)

Run this AFTER main.py has already built nascar.db.

Usage:   python fetch_races.py

Covers seasons 2020-2025.
"""

import requests
import sqlite3
import time
import re

DB_FILE    = "nascar.db"
START_YEAR = 2020
END_YEAR   = 2025
PAUSE      = 0.25   # seconds between API calls

BASE       = "http://sports.core.api.espn.com/v2/sports/racing/leagues/nascar-premier"

# ── API helpers ────────────────────────────────────────────────────────────────

def get(url):
    """GET with one retry."""
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

def get_all_pages(base_url):
    """Fetch all pages of a paginated ESPN endpoint."""
    results = []
    url = f"{base_url}&limit=100&page=1"
    while url:
        data = get(url)
        if not data:
            break
        results.extend(data.get("items", []))
        page_index = data.get("pageIndex", 1)
        page_count = data.get("pageCount", 1)
        if page_index < page_count:
            url = f"{base_url}&limit=100&page={page_index + 1}"
        else:
            url = None
        time.sleep(PAUSE)
    return results

def extract_stat(categories, stat_name):
    """Find a stat value by name across all stat categories."""
    for cat in categories:
        for s in cat.get("stats", []):
            if s.get("name") == stat_name:
                v = s.get("value")
                return int(v) if v is not None else None
    return None

# ── Database setup ─────────────────────────────────────────────────────────────

def setup_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            id        INTEGER PRIMARY KEY,
            full_name TEXT,
            city      TEXT,
            state     TEXT,
            length    REAL,
            shape     TEXT
        );

        CREATE TABLE IF NOT EXISTS races (
            id         TEXT PRIMARY KEY,
            year       INTEGER,
            name       TEXT,
            date       TEXT,
            track_id   INTEGER,
            race_num   INTEGER,
            FOREIGN KEY (track_id) REFERENCES tracks(id)
        );

        CREATE TABLE IF NOT EXISTS race_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id          TEXT NOT NULL,
            driver_id        INTEGER NOT NULL,
            finish_pos       INTEGER,
            start_pos        INTEGER,
            laps_completed   INTEGER,
            laps_led         INTEGER,
            championship_pts INTEGER,
            bonus_pts        INTEGER,
            penalty_pts      INTEGER,
            car_number       TEXT,
            manufacturer     TEXT,
            team             TEXT,
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (race_id, driver_id)
        );
    """)
    conn.commit()
    print("Race tables ready.")

# ── Track fetching (cached) ────────────────────────────────────────────────────

def fetch_venue(conn, venue_ref, venue_cache):
    """Fetch and cache a track/venue. Returns venue_id or None."""
    # Extract venue ID from the ref URL
    match = re.search(r"/venues/(\d+)", venue_ref)
    if not match:
        return None
    venue_id = int(match.group(1))

    if venue_id in venue_cache:
        return venue_id

    time.sleep(PAUSE)
    data = get(venue_ref)
    if not data or "error" in data:
        return None

    conn.execute("""
        INSERT OR IGNORE INTO tracks (id, full_name, city, state, length, shape)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        venue_id,
        data.get("fullName"),
        data.get("address", {}).get("city"),
        data.get("address", {}).get("state"),
        data.get("length"),
        data.get("shape"),
    ))
    conn.commit()
    venue_cache.add(venue_id)
    return venue_id

# ── Driver name lookup (reuses existing drivers table) ────────────────────────

def ensure_driver(conn, driver_id, year, known_drivers):
    """Make sure this driver exists in the drivers table."""
    if driver_id in known_drivers:
        return
    time.sleep(PAUSE)
    url = f"{BASE}/seasons/{year}/athletes/{driver_id}?lang=en&region=us"
    data = get(url)
    if data and "error" not in data:
        conn.execute("""
            INSERT OR IGNORE INTO drivers (id, first_name, last_name, display_name)
            VALUES (?, ?, ?, ?)
        """, (
            driver_id,
            data.get("firstName", ""),
            data.get("lastName", ""),
            data.get("displayName", ""),
        ))
        conn.commit()
    known_drivers.add(driver_id)

# ── Race result fetching ───────────────────────────────────────────────────────

def fetch_race(conn, event_ref, year, race_num, venue_cache, known_drivers):
    """Fetch one race event and store all results."""
    # 1. Get event details
    event = get(event_ref)
    if not event or "error" in event:
        return

    race_id   = event.get("id")
    race_name = event.get("name", "Unknown")
    race_date = event.get("date", "")[:10]   # YYYY-MM-DD

    # 2. Get track
    venues   = event.get("venues", [])
    track_id = None
    if venues:
        v_ref    = venues[0].get("$ref", "")
        track_id = fetch_venue(conn, v_ref, venue_cache) if v_ref else None

    # Skip if no competitions
    competitions = event.get("competitions", [])
    if not competitions:
        return

    # 3. Store the race
    conn.execute("""
        INSERT OR IGNORE INTO races (id, year, name, date, track_id, race_num)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (race_id, year, race_name, race_date, track_id, race_num))
    conn.commit()

    # 4. Get competition details (has finish/start order and car info)
    comp_ref  = competitions[0].get("$ref", "")
    comp_data = get(comp_ref) if comp_ref else None
    if not comp_data:
        return

    # Build a dict of driver_id -> {finish_pos, start_pos, car info}
    driver_positions = {}
    for c in comp_data.get("competitors", []):
        did = int(c.get("id", 0))
        if not did:
            continue
        v = c.get("vehicle", {})
        driver_positions[did] = {
            "finish_pos":  c.get("order"),
            "start_pos":   c.get("startOrder"),
            "car_number":  v.get("number"),
            "manufacturer":v.get("manufacturer"),
            "team":        v.get("team"),
            "stats_ref":   c.get("statistics", {}).get("$ref", ""),
        }

    time.sleep(PAUSE)

    # 5. For each driver, fetch race statistics
    saved = 0
    for driver_id, info in driver_positions.items():
        ensure_driver(conn, driver_id, year, known_drivers)

        stats_ref = info.get("stats_ref", "")
        laps_completed = laps_led = pts = bonus = penalty = None

        if stats_ref:
            time.sleep(PAUSE)
            stats_data = get(stats_ref)
            if stats_data and "error" not in stats_data:
                cats = stats_data.get("splits", {}).get("categories", [])
                laps_completed = extract_stat(cats, "lapsCompleted")
                laps_led       = extract_stat(cats, "lapsLead")
                pts            = extract_stat(cats, "championshipPts")
                bonus          = extract_stat(cats, "bonus")
                penalty        = extract_stat(cats, "penaltyPts")

        conn.execute("""
            INSERT OR IGNORE INTO race_results
                (race_id, driver_id, finish_pos, start_pos,
                 laps_completed, laps_led, championship_pts,
                 bonus_pts, penalty_pts, car_number, manufacturer, team)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race_id, driver_id,
            info["finish_pos"], info["start_pos"],
            laps_completed, laps_led, pts, bonus, penalty,
            info["car_number"], info["manufacturer"], info["team"],
        ))
        saved += 1

    conn.commit()
    print(f"    Saved: {race_name} ({race_date}) — {saved} drivers")
    time.sleep(PAUSE)

# ── Per-year fetch ─────────────────────────────────────────────────────────────

def fetch_year(conn, year, venue_cache, known_drivers):
    print(f"\n-- {year} ---------------------")

    # Get all event refs for this season
    url   = f"{BASE}/seasons/{year}/types/2/events?lang=en&region=us"
    items = get_all_pages(url)
    print(f"  Found {len(items)} events.")

    for race_num, item in enumerate(items, start=1):
        ref = item.get("$ref", "")
        if ref:
            fetch_race(conn, ref, year, race_num, venue_cache, known_drivers)

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  NASCAR Race Results Fetcher")
    print(f"  Pulling seasons {START_YEAR} - {END_YEAR}")
    print("  (this will take 30-60 minutes)")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)
    setup_tables(conn)

    # Load already-known driver IDs to avoid redundant lookups
    known_drivers = set(
        row[0] for row in conn.execute("SELECT id FROM drivers")
    )

    # Load already-processed races to allow resuming
    done_races = set(
        row[0] for row in conn.execute("SELECT id FROM races")
    )
    print(f"  Already have {len(done_races)} races in DB (will skip these).")

    venue_cache = set(
        row[0] for row in conn.execute("SELECT id FROM tracks")
    )

    for year in range(START_YEAR, END_YEAR + 1):
        fetch_year(conn, year, venue_cache, known_drivers)

    conn.close()

    print("\n" + "=" * 50)
    print("  Done! Race results added to nascar.db")
    print("  Run 'python query.py' to explore.")
    print("=" * 50)

if __name__ == "__main__":
    main()
