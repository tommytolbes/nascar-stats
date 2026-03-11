"""
Fetch League Teams
------------------
Loads all 42 Braswell Fantasy NASCAR team picks for a given segment
into the league_team_picks table.

Usage:   python fetch_teams.py
         python fetch_teams.py --year 2026 --segment 1

Teams are hardcoded per segment since the site structure isn't
easily parseable.  Update TEAMS_BY_SEGMENT when a new segment loads.
"""

import sqlite3
import sys
from difflib import SequenceMatcher

DB_FILE = "nascar.db"

# ── Team picks per segment ────────────────────────────────────────────────────
# Keys are (year, segment).  Driver names must be close to display_name in DB.

TEAMS_BY_SEGMENT = {
    (2026, 1): {
        "Alan Ross":           ["Denny Hamlin", "Christopher Bell", "Bubba Wallace", "Austin Cindric"],
        "Andrew Brown":        ["Christopher Bell", "Ryan Blaney", "Chris Buescher", "Erik Jones"],
        "Bob Francis":         ["William Byron", "Ryan Blaney", "Tyler Reddick", "Cole Custer"],
        "Brandon Braswell":    ["Christopher Bell", "Ryan Blaney", "Chris Buescher", "Austin Cindric"],
        "Brittany Hooper":     ["Ryan Blaney", "Chase Elliott", "Chris Buescher", "John Hunter Nemechek"],
        "Brooks Deaton":       ["William Byron", "Alex Bowman", "Carson Hocevar", "Brad Keselowski"],
        "Chris Burfield":      ["William Byron", "Ryan Blaney", "Chris Buescher", "Austin Cindric"],
        "Chris Faulk":         ["William Byron", "Christopher Bell", "Connor Zilisch", "John Hunter Nemechek"],
        "Chris Hodge":         ["William Byron", "Ross Chastain", "Carson Hocevar", "Kyle Busch"],
        "Chris House":         ["Ryan Blaney", "Shane Van Gisbergen", "Chris Buescher", "AJ Allmendinger"],
        "Dean Paxton":         ["Chase Elliott", "Shane Van Gisbergen", "Carson Hocevar", "Connor Zilisch"],
        "Dennis Williams":     ["Kyle Larson", "Shane Van Gisbergen", "Joey Logano", "Ricky Stenhouse Jr."],
        "Dustin Draughon":     ["Denny Hamlin", "Ryan Blaney", "Alex Bowman", "Austin Cindric"],
        "Dustin Reynolds":     ["Kyle Larson", "Denny Hamlin", "Ryan Preece", "Ty Dillon"],
        "Elisabeth Draughon":  ["Christopher Bell", "Chase Elliott", "Chris Buescher", "John Hunter Nemechek"],
        "Gary Bolen":          ["Ryan Blaney", "Joey Logano", "Alex Bowman", "Michael McDowell"],
        "Greg Camden":         ["Kyle Larson", "Ryan Blaney", "Kyle Busch", "Erik Jones"],
        "Greg Piel":           ["Ryan Blaney", "Chris Buescher", "Alex Bowman", "Michael McDowell"],
        "Jason Hollingsworth": ["William Byron", "Shane Van Gisbergen", "Ross Chastain", "Cody Ware"],
        "Jason House":         ["Christopher Bell", "Tyler Reddick", "Alex Bowman", "Ricky Stenhouse Jr."],
        "Jennifer Braswell":   ["William Byron", "Chase Elliott", "Tyler Reddick", "Austin Cindric"],
        "Jeremy Steele":       ["Christopher Bell", "Chase Elliott", "Tyler Reddick", "Austin Cindric"],
        "John Regan":          ["William Byron", "Ty Gibbs", "Chris Buescher", "Erik Jones"],
        "Justin Braswell":     ["Chase Elliott", "Tyler Reddick", "Joey Logano", "Austin Dillon"],
        "Keith Williams":      ["Chase Elliott", "Shane Van Gisbergen", "Brad Keselowski", "Daniel Suarez"],
        "Kevin Steele":        ["Denny Hamlin", "Ryan Blaney", "AJ Allmendinger", "John Hunter Nemechek"],
        "Kyle Zaepful":        ["Chase Briscoe", "Ryan Blaney", "Alex Bowman", "Austin Cindric"],
        "Larry Braswell":      ["Ryan Blaney", "Chase Elliott", "Michael McDowell", "Brad Keselowski"],
        "Manish Nagpal":       ["Tyler Reddick", "Ross Chastain", "Chris Buescher", "Michael McDowell"],
        "Mason Steele":        ["Kyle Larson", "Denny Hamlin", "Todd Gilliland", "Ricky Stenhouse Jr."],
        "Michael Klaus":       ["Denny Hamlin", "Chase Elliott", "Kyle Busch", "Ricky Stenhouse Jr."],
        "Mike Williams":       ["Christopher Bell", "Chase Elliott", "Tyler Reddick", "Austin Cindric"],
        "Miranda Brown":       ["Denny Hamlin", "Tyler Reddick", "Carson Hocevar", "Josh Berry"],
        "Nick Prytula":        ["Kyle Larson", "Christopher Bell", "Brad Keselowski", "Austin Dillon"],
        "Shawn White":         ["Kyle Larson", "Denny Hamlin", "Michael McDowell", "Austin Cindric"],
        "Steel Norman":        ["Ryan Blaney", "Joey Logano", "Todd Gilliland", "AJ Allmendinger"],
        "Stuart Adkins":       ["Denny Hamlin", "William Byron", "Carson Hocevar", "Ricky Stenhouse Jr."],
        "Thomas Tolbert":      ["Christopher Bell", "Ryan Blaney", "Tyler Reddick", "Austin Cindric"],
        "Tim Hodge":           ["Christopher Bell", "Ryan Blaney", "Alex Bowman", "Zane Smith"],
        "Tim Laney":           ["Kyle Larson", "Ryan Blaney", "Ross Chastain", "Cody Ware"],
        "Tucker Lovelace":     ["William Byron", "Christopher Bell", "Connor Zilisch", "Zane Smith"],
        "Valerie Jacobs":      ["Christopher Bell", "Shane Van Gisbergen", "Ryan Preece", "Kyle Busch"],
    },
}

# ── Database setup ─────────────────────────────────────────────────────────────

def setup_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS league_team_picks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            year       INTEGER NOT NULL,
            segment    INTEGER NOT NULL,
            team_name  TEXT    NOT NULL,
            driver_id  INTEGER NOT NULL,
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (year, segment, team_name, driver_id)
        );
    """)
    conn.commit()

# ── Driver name matcher (same logic as load_segment.py) ───────────────────────

def _similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_driver(scraped_name, all_drivers, threshold=0.78):
    best_id, best_name, best_score = None, None, 0.0
    for driver_id, display_name in all_drivers:
        score = _similarity(scraped_name, display_name)
        if score > best_score:
            best_score = score
            best_id    = driver_id
            best_name  = display_name
    if best_score >= threshold:
        return best_id, best_name, best_score
    return None

# ── Load teams into DB ─────────────────────────────────────────────────────────

def load_teams(conn, year, segment):
    key = (year, segment)
    if key not in TEAMS_BY_SEGMENT:
        print(f"No team data defined for {year} Segment {segment}.")
        print(f"Add an entry to TEAMS_BY_SEGMENT in fetch_teams.py.")
        return

    teams = TEAMS_BY_SEGMENT[key]
    all_drivers = conn.execute("SELECT id, display_name FROM drivers").fetchall()

    total_picks   = 0
    total_matched = 0
    unmatched     = []

    # Clear existing data for this year/segment so re-running is idempotent
    conn.execute(
        "DELETE FROM league_team_picks WHERE year = ? AND segment = ?",
        (year, segment)
    )

    for team_name, driver_names in teams.items():
        for raw_name in driver_names:
            total_picks += 1
            result = match_driver(raw_name, all_drivers)
            if result:
                driver_id, db_name, score = result
                conn.execute("""
                    INSERT OR IGNORE INTO league_team_picks
                        (year, segment, team_name, driver_id)
                    VALUES (?, ?, ?, ?)
                """, (year, segment, team_name, driver_id))
                total_matched += 1
            else:
                unmatched.append((team_name, raw_name))

    conn.commit()

    print(f"Teams loaded: {len(teams)} teams, {total_matched}/{total_picks} driver picks matched.")
    if unmatched:
        print(f"\nCould not match {len(unmatched)} driver name(s):")
        for team, name in unmatched:
            print(f"  [{team}] {name}")
        print("\nCheck spelling against the drivers table or lower the threshold.")

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    year    = 2026
    segment = 1

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--year"    and i + 1 < len(args): year    = int(args[i + 1])
        if arg == "--segment" and i + 1 < len(args): segment = int(args[i + 1])

    print("=" * 50)
    print(f"  Loading League Teams — {year} Segment {segment}")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)
    setup_tables(conn)
    load_teams(conn, year, segment)
    conn.close()

    print("\nDone. Run 'python build_fantasy.py' to compute team bonuses.")


if __name__ == "__main__":
    main()
