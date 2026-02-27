"""
Fantasy NASCAR Database Builder
---------------------------------
Creates four new tables in nascar.db:

  points_scale    : exact fantasy points per finishing position
  driver_salaries : driver cost per season/segment
  segments        : which races make up each 4-race segment
  fantasy_scores  : pre-calculated fantasy points per driver per race

Scoring used (stage points excluded for now):
  - Race pts       : position 1-41 (300 down to 5)
  - Qualifying pts : position 1-15 (75 down to 1)
  - Race leader    : +100 pts (driver who led the most laps)
  - Qual leader    : +25 pts  (pole sitter, start_pos = 1)

Run this after main.py and fetch_races.py have populated nascar.db.
Re-run any time new race data is added -- already-scored races are skipped.

Usage:   python build_fantasy.py
"""

import sqlite3

DB_FILE = "nascar.db"

# ── Points scales (exact values from league rules) ─────────────────────────────

RACE_PTS = {
     1: 300,  2: 250,  3: 220,  4: 200,  5: 180,
     6: 160,  7: 150,  8: 146,  9: 142, 10: 138,
    11: 134, 12: 130, 13: 126, 14: 122, 15: 118,
    16: 114, 17: 110, 18: 106, 19: 102, 20:  98,
    21:  94, 22:  90, 23:  86, 24:  82, 25:  78,
    26:  74, 27:  70, 28:  66, 29:  62, 30:  58,
    31:  54, 32:  50, 33:  45, 34:  40, 35:  35,
    36:  30, 37:  25, 38:  20, 39:  15, 40:  10,
    41:   5,
}

QUAL_PTS = {
     1: 75,  2: 50,  3: 45,  4: 40,  5: 35,
     6: 30,  7: 25,  8: 20,  9: 15, 10: 10,
    11:  8, 12:  6, 13:  4, 14:  2, 15:  1,
}

RACE_LEADER_BONUS = 100   # most laps led in race
QUAL_LEADER_BONUS  = 25   # pole sitter (start_pos = 1)

# ── 2026 Segment 1 driver salaries (Daytona 500 / Atlanta / COTA / Phoenix) ───
# Names must match display_name in the drivers table (case-insensitive lookup).

SALARIES_2026_SEG1 = {
    "Kyle Larson":           40,
    "Denny Hamlin":          38,
    "Chase Briscoe":         36,
    "William Byron":         35,
    "Christopher Bell":      34,
    "Ryan Blaney":           33,
    "Chase Elliott":         32,
    "Tyler Reddick":         30,
    "Shane Van Gisbergen":   29,
    "Ty Gibbs":              28,
    "Joey Logano":           27,
    "Ross Chastain":         26,
    "Chris Buescher":        25,
    "Bubba Wallace":         24,
    "Alex Bowman":           22,
    "Ryan Preece":           21,
    "Carson Hocevar":        20,
    "Connor Zilisch":        19,
    "Michael McDowell":      18,
    "Todd Gilliland":        17,
    "Kyle Busch":            16,
    "Brad Keselowski":       15,
    "Daniel Suarez":         14,
    "AJ Allmendinger":       13,
    "Josh Berry":            12,
    "Zane Smith":            11,
    "Austin Dillon":         10,
    "John Hunter Nemechek":   8,
    "Erik Jones":             6,
    "Ricky Stenhouse Jr.":    4,
    "Austin Cindric":         3,
    "Noah Gragson":           2,
    "Cole Custer":            1,
    "Riley Herbst":           1,
    "Ty Dillon":              1,
    "Cody Ware":              1,
}

# ── 2026 season segments (track names map to historical data) ─────────────────
# Each segment is 4 consecutive races.
# track_keywords are used to match track names in the races table.

SEGMENTS_2026 = [
    {
        "segment": 1,
        "races": [
            "Daytona 500",
            "Atlanta",
            "COTA",
            "Phoenix",
        ]
    },
    # Add future segments here as they are announced
]

# ── Database setup ─────────────────────────────────────────────────────────────

def setup_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS points_scale (
            position      INTEGER NOT NULL,
            scale_type    TEXT NOT NULL,   -- 'race' or 'qualifying'
            points        INTEGER NOT NULL,
            PRIMARY KEY (position, scale_type)
        );

        CREATE TABLE IF NOT EXISTS driver_salaries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id   INTEGER NOT NULL,
            year        INTEGER NOT NULL,
            segment     INTEGER NOT NULL,
            salary      INTEGER NOT NULL,
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (driver_id, year, segment)
        );

        CREATE TABLE IF NOT EXISTS segments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            year         INTEGER NOT NULL,
            segment_num  INTEGER NOT NULL,
            race_keyword TEXT NOT NULL,
            UNIQUE (year, segment_num, race_keyword)
        );

        CREATE TABLE IF NOT EXISTS fantasy_scores (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id          TEXT NOT NULL,
            driver_id        INTEGER NOT NULL,
            qualifying_pts   INTEGER NOT NULL DEFAULT 0,
            race_pts         INTEGER NOT NULL DEFAULT 0,
            qual_leader_bonus INTEGER NOT NULL DEFAULT 0,
            race_leader_bonus INTEGER NOT NULL DEFAULT 0,
            total_pts        INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (race_id, driver_id)
        );
    """)
    conn.commit()
    print("Tables ready.")

# ── Load points scale ──────────────────────────────────────────────────────────

def load_points_scale(conn):
    for pos, pts in RACE_PTS.items():
        conn.execute("""
            INSERT OR REPLACE INTO points_scale (position, scale_type, points)
            VALUES (?, 'race', ?)
        """, (pos, pts))

    for pos, pts in QUAL_PTS.items():
        conn.execute("""
            INSERT OR REPLACE INTO points_scale (position, scale_type, points)
            VALUES (?, 'qualifying', ?)
        """, (pos, pts))

    conn.commit()
    print(f"Points scale loaded: {len(RACE_PTS)} race positions, {len(QUAL_PTS)} qualifying positions.")

# ── Load segments ──────────────────────────────────────────────────────────────

def load_segments(conn):
    count = 0
    for seg in SEGMENTS_2026:
        for race_keyword in seg["races"]:
            conn.execute("""
                INSERT OR IGNORE INTO segments (year, segment_num, race_keyword)
                VALUES (?, ?, ?)
            """, (2026, seg["segment"], race_keyword))
            count += 1
    conn.commit()
    print(f"Segments loaded: {count} race slots.")

# ── Load driver salaries ───────────────────────────────────────────────────────

def load_salaries(conn):
    # Build a case-insensitive lookup of display_name -> driver_id
    rows = conn.execute("SELECT id, display_name FROM drivers").fetchall()
    name_map = {row[1].lower(): row[0] for row in rows}

    loaded = 0
    skipped = []

    for name, salary in SALARIES_2026_SEG1.items():
        driver_id = name_map.get(name.lower())
        if driver_id is None:
            skipped.append(name)
            continue

        conn.execute("""
            INSERT OR REPLACE INTO driver_salaries (driver_id, year, segment, salary)
            VALUES (?, 2026, 1, ?)
        """, (driver_id, salary))
        loaded += 1

    conn.commit()
    print(f"Salaries loaded: {loaded} drivers.")
    if skipped:
        print(f"  Could not match {len(skipped)} driver(s) - adding as unlinked:")
        for name in skipped:
            print(f"    - {name} (add manually or check spelling)")
            # Store with driver_id = -1 as a placeholder so salary isn't lost
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unmatched_salaries (
                    name    TEXT,
                    year    INTEGER,
                    segment INTEGER,
                    salary  INTEGER
                )
            """)
            conn.execute("""
                INSERT INTO unmatched_salaries (name, year, segment, salary)
                VALUES (?, 2026, 1, ?)
            """, (name, SALARIES_2026_SEG1[name]))
        conn.commit()

# ── Calculate fantasy scores ───────────────────────────────────────────────────

def calculate_fantasy_scores(conn):
    # Find all races that don't yet have fantasy scores calculated
    races = conn.execute("""
        SELECT r.id FROM races r
        WHERE NOT EXISTS (
            SELECT 1 FROM fantasy_scores fs WHERE fs.race_id = r.id
        )
    """).fetchall()

    print(f"Calculating fantasy scores for {len(races)} races...")

    for (race_id,) in races:
        # Get all drivers in this race
        results = conn.execute("""
            SELECT driver_id, finish_pos, start_pos, laps_led
            FROM race_results
            WHERE race_id = ?
        """, (race_id,)).fetchall()

        if not results:
            continue

        # Find the race leader (most laps led)
        max_laps = max((r[3] or 0) for r in results)
        leader_ids = set(
            r[0] for r in results if (r[3] or 0) == max_laps and max_laps > 0
        )

        for driver_id, finish_pos, start_pos, laps_led in results:
            race_pts  = RACE_PTS.get(finish_pos, 0) if finish_pos else 0
            qual_pts  = QUAL_PTS.get(start_pos, 0)  if start_pos else 0
            ql_bonus  = QUAL_LEADER_BONUS if start_pos == 1 else 0
            rl_bonus  = RACE_LEADER_BONUS if driver_id in leader_ids else 0
            total     = race_pts + qual_pts + ql_bonus + rl_bonus

            conn.execute("""
                INSERT OR IGNORE INTO fantasy_scores
                    (race_id, driver_id, qualifying_pts, race_pts,
                     qual_leader_bonus, race_leader_bonus, total_pts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (race_id, driver_id, qual_pts, race_pts, ql_bonus, rl_bonus, total))

        conn.commit()

    total_scored = conn.execute("SELECT COUNT(*) FROM fantasy_scores").fetchone()[0]
    print(f"Done. {total_scored} total scored driver-race entries in fantasy_scores.")

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Fantasy NASCAR Database Builder")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)

    setup_tables(conn)
    load_points_scale(conn)
    load_segments(conn)
    load_salaries(conn)
    calculate_fantasy_scores(conn)

    conn.close()
    print("\n" + "=" * 50)
    print("  Done! Run 'python query.py' to explore.")
    print("=" * 50)

if __name__ == "__main__":
    main()
