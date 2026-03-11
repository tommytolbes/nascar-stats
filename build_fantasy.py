"""
Fantasy NASCAR Database Builder
---------------------------------
Creates four new tables in nascar.db:

  points_scale    : exact fantasy points per finishing position
  driver_salaries : driver cost per season/segment
  segments        : which races make up each 4-race segment
  fantasy_scores  : pre-calculated fantasy points per driver per race

Scoring (individual):
  - Race pts       : position 1-41 (300 down to 5)
  - Qualifying pts : position 1-15 (75 down to 1)
  - Qual leader    : +25 pts  (pole sitter, start_pos = 1)
  - Stage pts      : same QUAL_PTS scale per stage (from fetch_stages.py data)

Team bonuses (per race, stored in team_race_bonuses):
  - Qualifying     : +25 to team with highest combined qualifying pts
  - Stage 1        : +25 to team with highest combined stage 1 pts
  - Stage 2        : +25 to team with highest combined stage 2 pts
  - Race           : +100 to team with highest combined race pts

Note: race_leader_bonus and stage_bonus columns in fantasy_scores are kept
for schema compatibility but are always 0 (these are team-only bonuses).

Run this after main.py and fetch_races.py have populated nascar.db.
Re-run any time new race data is added -- already-scored races are skipped.

Usage:   python build_fantasy.py
"""

import json
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
            stage_pts        INTEGER NOT NULL DEFAULT 0,
            stage_bonus      INTEGER NOT NULL DEFAULT 0,
            total_pts        INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id),
            UNIQUE (race_id, driver_id)
        );
    """)
    conn.commit()

    # Add new columns to existing DBs that pre-date them
    for col_def in [
        ("stage_pts",   "INTEGER NOT NULL DEFAULT 0"),
        ("stage_bonus", "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE fantasy_scores ADD COLUMN {col_def[0]} {col_def[1]}")
            conn.commit()
        except Exception:
            pass  # column already exists

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
            ql_bonus  = 0  # qual_leader_bonus is a TEAM-only bonus
            rl_bonus  = 0  # race_leader_bonus is a TEAM-only bonus
            total     = race_pts + qual_pts

            conn.execute("""
                INSERT OR IGNORE INTO fantasy_scores
                    (race_id, driver_id, qualifying_pts, race_pts,
                     qual_leader_bonus, race_leader_bonus, total_pts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (race_id, driver_id, qual_pts, race_pts, ql_bonus, rl_bonus, total))

        conn.commit()

    total_scored = conn.execute("SELECT COUNT(*) FROM fantasy_scores").fetchone()[0]
    print(f"Done. {total_scored} total scored driver-race entries in fantasy_scores.")

# ── Stage points update ────────────────────────────────────────────────────────

STAGE_BONUS = 25  # pts for winning a stage


def update_stage_pts(conn):
    """Apply stage points and bonuses from stage_results into fantasy_scores."""
    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stage_results'"
    ).fetchone()
    if not tbl:
        print("No stage_results table — skipping stage points.")
        return

    rows = conn.execute(
        "SELECT race_id, driver_id, stage1_pos, stage2_pos FROM stage_results"
    ).fetchall()

    if not rows:
        print("No stage results in DB — skipping stage points.")
        return

    updated = 0
    for race_id, driver_id, s1, s2 in rows:
        s1_pts      = QUAL_PTS.get(s1, 0) if s1 else 0
        s2_pts      = QUAL_PTS.get(s2, 0) if s2 else 0
        stage_pts   = s1_pts + s2_pts
        stage_bonus = 0  # stage win bonus is TEAM-only

        fs = conn.execute("""
            SELECT qualifying_pts, race_pts, qual_leader_bonus
            FROM fantasy_scores WHERE race_id=? AND driver_id=?
        """, (race_id, driver_id)).fetchone()
        if not fs:
            continue

        total = fs[0] + fs[1] + fs[2] + stage_pts

        conn.execute("""
            UPDATE fantasy_scores
            SET stage_pts=?, stage_bonus=?, total_pts=?
            WHERE race_id=? AND driver_id=?
        """, (stage_pts, stage_bonus, total, race_id, driver_id))
        updated += 1

    conn.commit()
    print(f"Stage points applied: {updated} driver-race entries updated.")


# ── Team bonus computation ─────────────────────────────────────────────────────
#
# For each race in each segment, the team whose 4 drivers combined for the
# most points in a given category wins a bonus:
#
#   Category          Bonus
#   ----------------  -----
#   Qualifying        +25
#   Stage 1           +25
#   Stage 2           +25
#   Race finish       +100
#
# Ties: all tied teams receive the bonus.
# A category is skipped (no bonus awarded) if the winning total is 0.

TEAM_QUAL_BONUS  = 25
TEAM_STAGE_BONUS = 25
TEAM_RACE_BONUS  = 100


def setup_team_bonus_table(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS team_race_bonuses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id    TEXT    NOT NULL,
            team_name  TEXT    NOT NULL,
            qual_bonus INTEGER NOT NULL DEFAULT 0,
            s1_bonus   INTEGER NOT NULL DEFAULT 0,
            s2_bonus   INTEGER NOT NULL DEFAULT 0,
            race_bonus INTEGER NOT NULL DEFAULT 0,
            UNIQUE (race_id, team_name)
        );
    """)
    conn.commit()


def compute_team_bonuses(conn):
    """
    Fill team_race_bonuses for every (year, segment) that has league_team_picks
    and completed fantasy_scores.  Safe to re-run; already-computed races are
    skipped via INSERT OR IGNORE.
    """
    # Need league_team_picks to exist
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='league_team_picks'"
    ).fetchone():
        print("No league_team_picks table — run fetch_teams.py first.")
        return

    # Each distinct year/segment in team picks
    seg_rows = conn.execute(
        "SELECT DISTINCT year, segment FROM league_team_picks ORDER BY year, segment"
    ).fetchall()

    if not seg_rows:
        print("league_team_picks is empty — run fetch_teams.py first.")
        return

    total_written = 0

    for year, segment in seg_rows:
        # Get track_ids for this segment (stored in segment_lineups by load_segment.py)
        sl_row = conn.execute(
            "SELECT track_ids FROM segment_lineups WHERE year=? AND segment=?",
            (year, segment)
        ).fetchone()
        if not sl_row:
            print(f"  No segment_lineups record for {year}/seg{segment} — skipping.")
            continue

        track_ids = json.loads(sl_row[0])
        ph        = ",".join("?" * len(track_ids))

        # Find all scored races for this segment (exclude non-points events)
        races = conn.execute(f"""
            SELECT DISTINCT r.id
            FROM races r
            WHERE r.track_id IN ({ph}) AND r.year = ?
              AND EXISTS (SELECT 1 FROM fantasy_scores fs WHERE fs.race_id = r.id)
              AND r.name NOT LIKE '%Duel%'
              AND r.name NOT LIKE '%Clash%'
              AND r.name NOT LIKE '%All-Star%'
              AND r.name NOT LIKE '%All Star%'
        """, (*track_ids, year)).fetchall()

        if not races:
            continue

        # Build team → [driver_ids] map
        teams = {}
        for team_name, driver_id in conn.execute(
            "SELECT team_name, driver_id FROM league_team_picks WHERE year=? AND segment=?",
            (year, segment)
        ).fetchall():
            teams.setdefault(team_name, []).append(driver_id)

        if not teams:
            continue

        for (race_id,) in races:
            # Skip if already computed for all teams
            existing = conn.execute(
                "SELECT COUNT(*) FROM team_race_bonuses WHERE race_id=?", (race_id,)
            ).fetchone()[0]
            if existing >= len(teams):
                continue

            # Pull stage positions for this race
            stage_map = {
                r[0]: (r[1], r[2])
                for r in conn.execute(
                    "SELECT driver_id, stage1_pos, stage2_pos FROM stage_results WHERE race_id=?",
                    (race_id,)
                ).fetchall()
            }

            # Pull qualifying_pts and race_pts from fantasy_scores
            fs_map = {
                r[0]: (r[1], r[2])
                for r in conn.execute(
                    "SELECT driver_id, qualifying_pts, race_pts FROM fantasy_scores WHERE race_id=?",
                    (race_id,)
                ).fetchall()
            }

            # Compute totals per team
            team_qual = {}
            team_s1   = {}
            team_s2   = {}
            team_race = {}

            for team_name, driver_ids in teams.items():
                q = s1 = s2 = rp = 0
                for did in driver_ids:
                    fs = fs_map.get(did)
                    if fs:
                        q  += fs[0]
                        rp += fs[1]
                    sr = stage_map.get(did)
                    if sr:
                        s1 += QUAL_PTS.get(sr[0], 0) if sr[0] else 0
                        s2 += QUAL_PTS.get(sr[1], 0) if sr[1] else 0
                team_qual[team_name] = q
                team_s1[team_name]   = s1
                team_s2[team_name]   = s2
                team_race[team_name] = rp

            max_qual = max(team_qual.values())
            max_s1   = max(team_s1.values())
            max_s2   = max(team_s2.values())
            max_race = max(team_race.values())

            for team_name in teams:
                qb  = TEAM_QUAL_BONUS  if max_qual > 0 and team_qual[team_name] == max_qual else 0
                s1b = TEAM_STAGE_BONUS if max_s1   > 0 and team_s1[team_name]   == max_s1   else 0
                s2b = TEAM_STAGE_BONUS if max_s2   > 0 and team_s2[team_name]   == max_s2   else 0
                rb  = TEAM_RACE_BONUS  if max_race > 0 and team_race[team_name] == max_race  else 0

                conn.execute("""
                    INSERT OR IGNORE INTO team_race_bonuses
                        (race_id, team_name, qual_bonus, s1_bonus, s2_bonus, race_bonus)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (race_id, team_name, qb, s1b, s2b, rb))
                total_written += 1

        conn.commit()

    print(f"Team bonuses computed: {total_written} team-race entries written.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Fantasy NASCAR Database Builder")
    print("=" * 50)

    conn = sqlite3.connect(DB_FILE)

    setup_tables(conn)
    setup_team_bonus_table(conn)
    load_points_scale(conn)
    load_segments(conn)
    load_salaries(conn)
    calculate_fantasy_scores(conn)
    update_stage_pts(conn)
    compute_team_bonuses(conn)

    conn.close()
    print("\n" + "=" * 50)
    print("  Done! Run 'python query.py' to explore.")
    print("=" * 50)

if __name__ == "__main__":
    main()
