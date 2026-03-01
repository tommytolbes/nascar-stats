"""
NASCAR Database Explorer
-------------------------
Run this after main.py to explore the data you've collected.

Usage:   python query.py

The active segment is controlled by segment.json.
Run 'python load_segment.py' at the start of each new segment.
"""

import sqlite3
import itertools
import json
import os

DB_FILE     = "nascar.db"
CONFIG_FILE = "segment.json"


def load_config():
    """Load the active segment config from segment.json."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    # Fallback if segment.json is missing
    return {
        "year":        2026,
        "segment":     1,
        "track_ids":   [1, 18, 253, 16],
        "track_names": [
            "Daytona International Speedway",
            "Atlanta Motor Speedway",
            "Circuit of the Americas",
            "Phoenix International Raceway",
        ],
    }


def run(conn, label, sql, params=()):
    print(f"\n{'-'*55}")
    print(f"  {label}")
    print(f"{'-'*55}")
    cursor = conn.execute(sql, params)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        print("  (no results)")
        return
    widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    header = "  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main():
    conn = sqlite3.connect(DB_FILE)
    cfg  = load_config()
    yr   = cfg["year"]
    seg  = cfg["segment"]
    tids = cfg["track_ids"]
    tnames = cfg["track_names"]

    print(f"\n{'='*55}")
    print(f"  Active: {yr} Segment {seg}")
    print(f"  Tracks: {', '.join(tnames)}")
    print(f"{'='*55}")

    # ── Historical Queries ──────────────────────────────────────────────────────

    # 1. Database summary
    run(conn, "Database Summary",
        """
        SELECT
            (SELECT COUNT(*) FROM drivers)                   AS total_drivers,
            (SELECT COUNT(DISTINCT year) FROM season_standings) AS total_seasons,
            (SELECT COUNT(*) FROM season_standings)          AS total_records,
            (SELECT COUNT(*) FROM races)                     AS total_races,
            (SELECT COUNT(*) FROM race_results)              AS total_results
        """)

    # 2. Most wins in a single season (top 15)
    run(conn, "Most Wins in a Single Season (Top 15)",
        """
        SELECT d.display_name, s.year, s.wins, s.top5, s.top10, s.championship_pts
        FROM season_standings s
        JOIN drivers d ON d.id = s.driver_id
        ORDER BY s.wins DESC
        LIMIT 15
        """)

    # 3. Career wins all-time (top 15)
    run(conn, "All-Time Career Wins (Top 15, 2000-2024)",
        """
        SELECT d.display_name,
               SUM(s.wins)   AS career_wins,
               SUM(s.starts) AS career_starts,
               COUNT(s.year) AS seasons,
               MIN(s.year)   AS first_year,
               MAX(s.year)   AS last_year
        FROM season_standings s
        JOIN drivers d ON d.id = s.driver_id
        GROUP BY d.id
        ORDER BY career_wins DESC
        LIMIT 15
        """)

    # 4. Championship winners per year
    run(conn, "NASCAR Cup Series Champions (2000-2024)",
        """
        SELECT s.year, d.display_name, s.wins, s.championship_pts
        FROM season_standings s
        JOIN drivers d ON d.id = s.driver_id
        WHERE s.rank = 1
        ORDER BY s.year DESC
        """)

    # 5. Most laps led in a single season
    run(conn, "Most Laps Led in a Single Season (Top 10)",
        """
        SELECT d.display_name, s.year, s.laps_led, s.wins
        FROM season_standings s
        JOIN drivers d ON d.id = s.driver_id
        ORDER BY s.laps_led DESC
        LIMIT 10
        """)

    # 6. Most poles in a single season
    run(conn, "Most Poles in a Single Season (Top 10)",
        """
        SELECT d.display_name, s.year, s.poles, s.wins
        FROM season_standings s
        JOIN drivers d ON d.id = s.driver_id
        ORDER BY s.poles DESC
        LIMIT 10
        """)

    # ── Fantasy NASCAR Queries ──────────────────────────────────────────────────

    # 7. Driver salaries for the active segment
    run(conn, f"{yr} Segment {seg} - Driver Salaries",
        """
        SELECT d.display_name, ds.salary
        FROM driver_salaries ds
        JOIN drivers d ON d.id = ds.driver_id
        WHERE ds.year = ? AND ds.segment = ?
        ORDER BY ds.salary DESC
        """, (yr, seg))

    # 8. Recent form - avg fantasy pts over the last 8 races
    run(conn, "Recent Form - Avg Fantasy Pts (Last 8 Races)",
        """
        WITH last8 AS (
            SELECT r.id AS race_id
            FROM races r
            ORDER BY r.date DESC
            LIMIT 8
        )
        SELECT
            d.display_name,
            ds.salary,
            ROUND(AVG(fs.total_pts), 1)              AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts) / ds.salary, 2)  AS pts_per_dollar,
            MIN(fs.total_pts)                        AS worst,
            MAX(fs.total_pts)                        AS best
        FROM fantasy_scores fs
        JOIN last8 ON last8.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id
        ORDER BY avg_fantasy_pts DESC
        """, (yr, seg))

    # 9-12. Per-track history for each track in the active segment
    for tid, tname in zip(tids, tnames):
        run(conn, f"Track History - {tname}",
            """
            SELECT
                d.display_name,
                ds.salary,
                COUNT(*)                              AS starts,
                ROUND(AVG(fs.total_pts), 1)           AS avg_fantasy_pts,
                ROUND(AVG(fs.total_pts)/ds.salary, 2) AS pts_per_dollar,
                MAX(fs.total_pts)                     AS best_score,
                ROUND(AVG(rr.finish_pos), 1)          AS avg_finish
            FROM fantasy_scores fs
            JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
            JOIN races r ON r.id = fs.race_id
            JOIN tracks t ON t.id = r.track_id
            JOIN drivers d ON d.id = fs.driver_id
            JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
                AND ds.year = ? AND ds.segment = ?
            WHERE t.id = ?
            GROUP BY fs.driver_id
            HAVING starts >= 1
            ORDER BY avg_fantasy_pts DESC
            """, (yr, seg, tid))

    # 13. Overall value across all segment tracks combined
    placeholders = ",".join("?" * len(tids))
    run(conn, f"Segment {seg} - Best Overall Value (All Tracks Combined)",
        f"""
        WITH seg_tracks AS (
            SELECT r.id AS race_id
            FROM races r
            JOIN tracks t ON t.id = r.track_id
            WHERE t.id IN ({placeholders})
        )
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                                  AS historical_starts,
            ROUND(AVG(fs.total_pts), 1)               AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts) / ds.salary, 2)   AS pts_per_dollar
        FROM fantasy_scores fs
        JOIN seg_tracks ON seg_tracks.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id
        HAVING historical_starts >= 2
        ORDER BY pts_per_dollar DESC
        LIMIT 20
        """, (*tids, yr, seg))

    # 14. Track type specialists - floor, ceiling, avg by track type
    run(conn, "Driver Avg Fantasy Pts by Track Type (min 3 starts per type)",
        """
        SELECT
            d.display_name,
            t.track_type,
            COUNT(*)                            AS starts,
            ROUND(AVG(fs.total_pts), 1)         AS avg_fantasy_pts,
            MIN(fs.total_pts)                   AS floor,
            MAX(fs.total_pts)                   AS ceiling
        FROM fantasy_scores fs
        JOIN races r ON r.id = fs.race_id
        JOIN tracks t ON t.id = r.track_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id, t.track_type
        HAVING starts >= 3
        ORDER BY t.track_type, avg_fantasy_pts DESC
        """, (yr, seg))

    # 15. Team optimizer
    fantasy_optimizer(conn, yr, seg, tids)

    conn.close()
    print(f"\n{'-'*55}")
    print("  Tip: open nascar.db with 'DB Browser for SQLite'")
    print("  (free at sqlitebrowser.org) to browse data visually.")
    print(f"  To load a new segment: python load_segment.py")
    print(f"{'-'*55}\n")


def fantasy_optimizer(conn, yr, seg, tids):
    """Find the top 5 four-driver combos under $100 for the active segment."""
    print(f"\n{'-'*55}")
    print(f"  Team Optimizer - Best 4-Driver Combos Under $100")
    print(f"{'-'*55}")

    placeholders = ",".join("?" * len(tids))
    rows = conn.execute(f"""
        WITH seg_tracks AS (
            SELECT r.id AS race_id
            FROM races r
            JOIN tracks t ON t.id = r.track_id
            WHERE t.id IN ({placeholders})
        )
        SELECT
            d.display_name,
            ds.salary,
            ROUND(AVG(fs.total_pts), 1) AS avg_pts
        FROM fantasy_scores fs
        JOIN seg_tracks ON seg_tracks.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id
        HAVING COUNT(*) >= 2
        ORDER BY avg_pts DESC
    """, (*tids, yr, seg)).fetchall()

    if not rows:
        print("  (no data available)")
        return

    best = []
    for combo in itertools.combinations(rows, 4):
        total_salary = sum(c[1] for c in combo)
        if total_salary > 100:
            continue
        total_pts = sum(c[2] for c in combo)
        best.append((total_pts, total_salary, combo))

    best.sort(reverse=True)

    if not best:
        print("  No valid combinations found under $100.")
        return

    for rank, (pts, salary, combo) in enumerate(best[:5], 1):
        names = " / ".join(c[0] for c in combo)
        costs = " + ".join(f"${c[1]}" for c in combo)
        print(f"\n  #{rank}  {round(pts,1)} avg pts  |  ${salary} total  |  ${100-salary} leftover")
        print(f"       {names}")
        print(f"       {costs}")


if __name__ == "__main__":
    main()
