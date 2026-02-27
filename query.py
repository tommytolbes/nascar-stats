"""
NASCAR Database Explorer
-------------------------
Run this after main.py to explore the data you've collected.

Usage:   python query.py
"""

import sqlite3
import itertools

DB_FILE = "nascar.db"

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
    # Column widths
    widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    header = "  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  " + "  ".join("-" * w for w in widths))  # noqa: separator line
    for row in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))

def main():
    conn = sqlite3.connect(DB_FILE)

    # 1. How many drivers and seasons do we have?
    run(conn, "Database Summary",
        """
        SELECT
            (SELECT COUNT(*) FROM drivers)          AS total_drivers,
            (SELECT COUNT(DISTINCT year) FROM season_standings) AS total_seasons,
            (SELECT COUNT(*) FROM season_standings) AS total_records
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

    # 7. 2026 Segment 1 driver salaries
    run(conn, "2026 Segment 1 - Driver Salaries",
        """
        SELECT d.display_name, ds.salary
        FROM driver_salaries ds
        JOIN drivers d ON d.id = ds.driver_id
        WHERE ds.year = 2026 AND ds.segment = 1
        ORDER BY ds.salary DESC
        """)

    # 8. Recent form - avg fantasy pts per driver over last 8 races of 2025
    run(conn, "Recent Form - Avg Fantasy Pts (Last 8 Races of 2025)",
        """
        WITH last8 AS (
            SELECT r.id AS race_id
            FROM races r
            WHERE r.year = 2025
            ORDER BY r.date DESC
            LIMIT 8
        )
        SELECT
            d.display_name,
            ds.salary,
            ROUND(AVG(fs.total_pts), 1)        AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts) / ds.salary, 2) AS pts_per_dollar,
            MIN(fs.total_pts)                  AS worst,
            MAX(fs.total_pts)                  AS best
        FROM fantasy_scores fs
        JOIN last8 ON last8.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        GROUP BY fs.driver_id
        ORDER BY avg_fantasy_pts DESC
        """)

    # 9. Track history - Daytona (superspeedway)
    run(conn, "Track History - Daytona (2020-2025)",
        """
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                            AS starts,
            ROUND(AVG(fs.total_pts), 1)         AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts)/ds.salary, 2) AS pts_per_dollar,
            MAX(fs.total_pts)                   AS best_score,
            ROUND(AVG(rr.finish_pos), 1)        AS avg_finish
        FROM fantasy_scores fs
        JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
        JOIN races r ON r.id = fs.race_id
        JOIN tracks t ON t.id = r.track_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        WHERE t.full_name LIKE '%Daytona%'
        GROUP BY fs.driver_id
        HAVING starts >= 1
        ORDER BY avg_fantasy_pts DESC
        """)

    # 10. Track history - Atlanta
    run(conn, "Track History - Atlanta (2020-2025)",
        """
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                            AS starts,
            ROUND(AVG(fs.total_pts), 1)         AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts)/ds.salary, 2) AS pts_per_dollar,
            MAX(fs.total_pts)                   AS best_score,
            ROUND(AVG(rr.finish_pos), 1)        AS avg_finish
        FROM fantasy_scores fs
        JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
        JOIN races r ON r.id = fs.race_id
        JOIN tracks t ON t.id = r.track_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        WHERE t.full_name LIKE '%Atlanta%'
        GROUP BY fs.driver_id
        HAVING starts >= 1
        ORDER BY avg_fantasy_pts DESC
        """)

    # 11. Track history - COTA (road course)
    run(conn, "Track History - COTA (2020-2025)",
        """
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                            AS starts,
            ROUND(AVG(fs.total_pts), 1)         AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts)/ds.salary, 2) AS pts_per_dollar,
            MAX(fs.total_pts)                   AS best_score,
            ROUND(AVG(rr.finish_pos), 1)        AS avg_finish
        FROM fantasy_scores fs
        JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
        JOIN races r ON r.id = fs.race_id
        JOIN tracks t ON t.id = r.track_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        WHERE t.full_name LIKE '%Texas%' OR t.full_name LIKE '%Circuit of%'
        GROUP BY fs.driver_id
        HAVING starts >= 1
        ORDER BY avg_fantasy_pts DESC
        """)

    # 12. Track history - Phoenix
    run(conn, "Track History - Phoenix (2020-2025)",
        """
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                            AS starts,
            ROUND(AVG(fs.total_pts), 1)         AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts)/ds.salary, 2) AS pts_per_dollar,
            MAX(fs.total_pts)                   AS best_score,
            ROUND(AVG(rr.finish_pos), 1)        AS avg_finish
        FROM fantasy_scores fs
        JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
        JOIN races r ON r.id = fs.race_id
        JOIN tracks t ON t.id = r.track_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        WHERE t.full_name LIKE '%Phoenix%'
        GROUP BY fs.driver_id
        HAVING starts >= 1
        ORDER BY avg_fantasy_pts DESC
        """)

    # 13. Overall segment 1 value score (avg across all 4 track types)
    run(conn, "Segment 1 - Best Overall Value (All 4 Tracks Combined)",
        """
        WITH seg_tracks AS (
            SELECT r.id AS race_id
            FROM races r
            JOIN tracks t ON t.id = r.track_id
            WHERE (
                t.full_name LIKE '%Daytona%' OR
                t.full_name LIKE '%Atlanta%' OR
                t.full_name LIKE '%Texas%'   OR
                t.full_name LIKE '%Circuit of%' OR
                t.full_name LIKE '%Phoenix%'
            )
        )
        SELECT
            d.display_name,
            ds.salary,
            COUNT(*)                                 AS historical_starts,
            ROUND(AVG(fs.total_pts), 1)              AS avg_fantasy_pts,
            ROUND(AVG(fs.total_pts) / ds.salary, 2)  AS pts_per_dollar
        FROM fantasy_scores fs
        JOIN seg_tracks ON seg_tracks.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        GROUP BY fs.driver_id
        HAVING historical_starts >= 2
        ORDER BY pts_per_dollar DESC
        LIMIT 20
        """)

    # 14. Team optimizer - best 4-driver combo under $100
    fantasy_optimizer(conn)

    conn.close()
    print(f"\n{'-'*55}")
    print("  Tip: open nascar.db with 'DB Browser for SQLite'")
    print("  (free download at sqlitebrowser.org) to browse visually.")
    print(f"{'-'*55}\n")


def fantasy_optimizer(conn):
    """
    Finds the top 5 four-driver combinations under $100
    ranked by combined average fantasy points across Segment 1 track types.
    """
    print(f"\n{'-'*55}")
    print("  Team Optimizer - Best 4-Driver Combos Under $100")
    print(f"{'-'*55}")

    # Pull drivers with salary + avg pts across segment 1 track types
    rows = conn.execute("""
        WITH seg_tracks AS (
            SELECT r.id AS race_id
            FROM races r
            JOIN tracks t ON t.id = r.track_id
            WHERE (
                t.full_name LIKE '%Daytona%' OR
                t.full_name LIKE '%Atlanta%' OR
                t.full_name LIKE '%Texas%'   OR
                t.full_name LIKE '%Circuit of%' OR
                t.full_name LIKE '%Phoenix%'
            )
        )
        SELECT
            d.display_name,
            ds.salary,
            ROUND(AVG(fs.total_pts), 1) AS avg_pts
        FROM fantasy_scores fs
        JOIN seg_tracks ON seg_tracks.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = 2026 AND ds.segment = 1
        GROUP BY fs.driver_id
        HAVING COUNT(*) >= 2
        ORDER BY avg_pts DESC
    """).fetchall()

    if not rows:
        print("  (no data available)")
        return

    # Evaluate all 4-driver combos
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
