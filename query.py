"""
NASCAR Database Explorer
-------------------------
Run this after main.py to explore the data you've collected.

Usage:   python query.py
"""

import sqlite3

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

    conn.close()
    print(f"\n{'-'*55}")
    print("  Tip: open nascar.db with 'DB Browser for SQLite'")
    print("  (free download at sqlitebrowser.org) to browse visually.")
    print(f"{'-'*55}\n")

if __name__ == "__main__":
    main()
