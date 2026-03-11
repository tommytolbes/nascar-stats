"""
NASCAR Fantasy Report Generator
---------------------------------
Generates index.html -- a webpage version of your fantasy analysis.
"""

import sqlite3
import json
import os
import itertools
import datetime
import requests
from html.parser import HTMLParser

DB_FILE     = "nascar.db"
CONFIG_FILE = "segment.json"
OUTPUT_FILE = "index.html"

USER_TEAM     = "Thomas Tolbert"
STANDINGS_URL = "https://www.braswellsfantasynascar.com/standings.html"
STANDINGS_TTL = 6 * 3600


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "year": 2026, "segment": 1,
        "track_ids":   [1, 18, 253, 16],
        "track_names": ["Daytona International Speedway", "Atlanta Motor Speedway",
                        "Circuit of the Americas", "Phoenix International Raceway"],
    }


def q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def setup_segment_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS segment_lineups (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            year      INTEGER NOT NULL,
            segment   INTEGER NOT NULL,
            driver_1  INTEGER,
            driver_2  INTEGER,
            driver_3  INTEGER,
            driver_4  INTEGER,
            track_ids TEXT,
            UNIQUE(year, segment)
        );
        CREATE TABLE IF NOT EXISTS segment_optimal_lineups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL, segment INTEGER NOT NULL,
            driver_1 INTEGER, driver_2 INTEGER, driver_3 INTEGER, driver_4 INTEGER,
            salary_total INTEGER, segment_points REAL,
            UNIQUE(year, segment)
        );
        CREATE TABLE IF NOT EXISTS league_standings_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rank INTEGER, name TEXT, points TEXT,
            behind TEXT, weekly TEXT, last_week TEXT, last_updated TEXT
        );
    """)
    conn.commit()


def get_recent_form(conn, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    return q(conn, """
        WITH last8 AS (
            SELECT r.id AS race_id FROM races r
            WHERE EXISTS (SELECT 1 FROM race_results rr WHERE rr.race_id = r.id)
            ORDER BY r.date DESC LIMIT 8
        )
        SELECT d.display_name AS driver, ds.salary,
               ROUND(AVG(fs.total_pts), 1)             AS avg_pts,
               ROUND(AVG(fs.total_pts) / ds.salary, 2) AS pts_per_dollar,
               MIN(fs.total_pts) AS floor,
               MAX(fs.total_pts) AS ceiling
        FROM fantasy_scores fs
        JOIN last8 ON last8.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id
        ORDER BY avg_pts DESC
    """, (yr, seg))


def get_track_histories(conn, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    result = {}
    for tid, tname in zip(cfg["track_ids"], cfg["track_names"]):
        result[tname] = q(conn, """
            SELECT d.display_name AS driver, ds.salary,
                   COUNT(*) AS starts,
                   ROUND(AVG(fs.total_pts), 1)             AS avg_pts,
                   ROUND(AVG(fs.total_pts)/ds.salary, 2)   AS pts_per_dollar,
                   MAX(fs.total_pts)                       AS best_score,
                   ROUND(AVG(rr.finish_pos), 1)            AS avg_finish
            FROM fantasy_scores fs
            JOIN race_results rr ON rr.race_id = fs.race_id AND rr.driver_id = fs.driver_id
            JOIN races r ON r.id = fs.race_id
            JOIN tracks t ON t.id = r.track_id
            JOIN drivers d ON d.id = fs.driver_id
            JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
                AND ds.year = ? AND ds.segment = ?
            WHERE t.id = ?
            GROUP BY fs.driver_id HAVING starts >= 1
            ORDER BY avg_pts DESC
        """, (yr, seg, tid))
    return result


def get_race_results(conn, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    result = {}
    for tid, tname in zip(cfg["track_ids"], cfg["track_names"]):
        latest = conn.execute("""
            SELECT id FROM races
            WHERE track_id = ? AND year = ?
              AND EXISTS (SELECT 1 FROM race_results rr WHERE rr.race_id = races.id)
            ORDER BY date DESC LIMIT 1
        """, (tid, yr)).fetchone()
        if not latest:
            result[tname] = []
            continue
        result[tname] = q(conn, """
            WITH hist AS (
                SELECT rr.driver_id,
                       ROUND(AVG(CAST(rr.finish_pos AS REAL)), 1) AS hist_avg
                FROM race_results rr
                JOIN races r ON r.id = rr.race_id
                WHERE r.track_id = ? AND r.year < ?
                GROUP BY rr.driver_id
            )
            SELECT d.display_name AS driver, ds.salary,
                   rr.finish_pos, rr.start_pos, h.hist_avg,
                   CASE WHEN h.hist_avg IS NOT NULL
                        THEN ROUND(h.hist_avg - rr.finish_pos, 1)
                        ELSE NULL END AS plus_minus
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN driver_salaries ds ON ds.driver_id = d.id
                AND ds.year = ? AND ds.segment = ?
            LEFT JOIN hist h ON h.driver_id = rr.driver_id
            WHERE rr.race_id = ?
            ORDER BY rr.finish_pos
        """, (tid, yr, yr, seg, latest[0]))
    return result


def _driver_name(conn, driver_id):
    row = conn.execute("SELECT display_name FROM drivers WHERE id = ?", (driver_id,)).fetchone()
    return row[0] if row else "Driver " + str(driver_id)


def _fmt(n):
    """Format a number with commas. Drops .0 decimals."""
    if isinstance(n, float) and n == int(n):
        return f"{int(n):,}"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{int(n):,}"


def get_user_lineup(conn, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    row = conn.execute(
        "SELECT driver_1, driver_2, driver_3, driver_4, track_ids"
        " FROM segment_lineups WHERE year = ? AND segment = ?", (yr, seg)
    ).fetchone()
    if not row:
        return None

    driver_ids = [row[0], row[1], row[2], row[3]]
    track_ids  = json.loads(row[4]) if row[4] else cfg["track_ids"]

    drivers = []
    for did in driver_ids:
        if did is None:
            continue
        name = _driver_name(conn, did)
        sal  = conn.execute(
            "SELECT salary FROM driver_salaries WHERE driver_id = ? AND year = ? AND segment = ?",
            (did, yr, seg)
        ).fetchone()
        salary = sal[0] if sal else 0

        ph = ",".join("?" * len(track_ids))
        pts_row = conn.execute(
            "SELECT COALESCE(SUM(fs.total_pts), 0) FROM fantasy_scores fs"
            " JOIN races r ON r.id = fs.race_id"
            " WHERE fs.driver_id = ? AND r.track_id IN (" + ph + ") AND r.year = ? AND r.name NOT LIKE '%Duel%' AND r.name NOT LIKE '%Clash%' AND r.name NOT LIKE '%All-Star%' AND r.name NOT LIKE '%All Star%'",
            (did, *track_ids, yr)
        ).fetchone()
        pts = round(pts_row[0], 1) if pts_row else 0
        drivers.append({"name": name, "salary": salary, "pts": pts})

    total_salary = sum(d["salary"] for d in drivers)
    driver_pts   = round(sum(d["pts"] for d in drivers), 1)
    team_bonus   = get_team_bonus_total(conn, USER_TEAM, track_ids, yr)
    total_pts    = driver_pts   # team bonus shown separately; not baked into score

    ph = ",".join("?" * len(track_ids))
    races_done = conn.execute(
        "SELECT COUNT(DISTINCT r.id) FROM races r"
        " WHERE r.track_id IN (" + ph + ") AND r.year = ?"
        " AND EXISTS (SELECT 1 FROM race_results rr"
        "             WHERE rr.race_id = r.id AND rr.finish_pos IS NOT NULL)"
        " AND r.name NOT LIKE '%Duel%' AND r.name NOT LIKE '%Clash%' AND r.name NOT LIKE '%All-Star%' AND r.name NOT LIKE '%All Star%'",
        (*track_ids, yr)
    ).fetchone()[0]

    return {
        "drivers":      drivers,
        "total_salary": total_salary,
        "driver_pts":   driver_pts,
        "team_bonus":   team_bonus,
        "total_pts":    total_pts,
        "races_done":   races_done,
        "total_races":  len(track_ids),
    }


def get_prev_lineup(conn, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    if seg <= 1:
        return None
    prev_seg = seg - 1
    row = conn.execute(
        "SELECT driver_1, driver_2, driver_3, driver_4, track_ids"
        " FROM segment_lineups WHERE year = ? AND segment = ?", (yr, prev_seg)
    ).fetchone()
    if not row:
        return None

    driver_ids = [row[0], row[1], row[2], row[3]]
    track_ids  = json.loads(row[4]) if row[4] else []

    drivers = []
    for did in driver_ids:
        if did is None:
            continue
        name = _driver_name(conn, did)
        sal  = conn.execute(
            "SELECT salary FROM driver_salaries WHERE driver_id = ? AND year = ? AND segment = ?",
            (did, yr, prev_seg)
        ).fetchone()
        salary = sal[0] if sal else 0

        pts = 0
        if track_ids:
            ph = ",".join("?" * len(track_ids))
            pts_row = conn.execute(
                "SELECT COALESCE(SUM(fs.total_pts), 0) FROM fantasy_scores fs"
                " JOIN races r ON r.id = fs.race_id"
                " WHERE fs.driver_id = ? AND r.track_id IN (" + ph + ") AND r.year = ? AND r.name NOT LIKE '%Duel%' AND r.name NOT LIKE '%Clash%' AND r.name NOT LIKE '%All-Star%' AND r.name NOT LIKE '%All Star%'",
                (did, *track_ids, yr)
            ).fetchone()
            pts = round(pts_row[0], 1) if pts_row else 0

        drivers.append({"name": name, "salary": salary, "pts": pts})

    driver_pts   = round(sum(d["pts"] for d in drivers), 1)
    team_bonus   = get_team_bonus_total(conn, USER_TEAM, track_ids, yr)

    return {
        "drivers":      drivers,
        "total_salary": sum(d["salary"] for d in drivers),
        "driver_pts":   driver_pts,
        "team_bonus":   team_bonus,
        "total_pts":    driver_pts,  # team bonus shown separately; not baked into score
        "segment":      prev_seg,
        "track_ids":    track_ids,
    }


def get_or_compute_optimal(conn, year, segment, track_ids):
    if not track_ids:
        return None

    cached = conn.execute(
        "SELECT driver_1, driver_2, driver_3, driver_4, salary_total, segment_points"
        " FROM segment_optimal_lineups WHERE year = ? AND segment = ?",
        (year, segment)
    ).fetchone()
    if cached:
        driver_ids = [cached[0], cached[1], cached[2], cached[3]]
        def _sal(did):
            row = conn.execute(
                "SELECT salary FROM driver_salaries WHERE driver_id=? AND year=? AND segment=?",
                (did, year, segment)
            ).fetchone()
            return row[0] if row else 0
        return {
            "drivers":      [{"name": _driver_name(conn, did), "salary": _sal(did)} for did in driver_ids],
            "total_salary": cached[4],
            "total_pts":    round(cached[5], 1),
        }

    ph = ",".join("?" * len(track_ids))
    complete = conn.execute(
        "SELECT COUNT(DISTINCT r.track_id) FROM races r"
        " WHERE r.track_id IN (" + ph + ") AND r.year = ?"
        " AND EXISTS (SELECT 1 FROM race_results rr"
        "             WHERE rr.race_id = r.id AND rr.finish_pos IS NOT NULL)"
        " AND r.name NOT LIKE '%Duel%' AND r.name NOT LIKE '%Clash%' AND r.name NOT LIKE '%All-Star%' AND r.name NOT LIKE '%All Star%'",
        (*track_ids, year)
    ).fetchone()[0]

    if complete < len(track_ids):
        return None

    rows = conn.execute(
        "SELECT fs.driver_id, d.display_name, ds.salary, SUM(fs.total_pts) AS seg_pts"
        " FROM fantasy_scores fs"
        " JOIN races r ON r.id = fs.race_id"
        " JOIN drivers d ON d.id = fs.driver_id"
        " JOIN driver_salaries ds ON ds.driver_id = fs.driver_id"
        "     AND ds.year = ? AND ds.segment = ?"
        " WHERE r.track_id IN (" + ph + ") AND r.year = ? AND r.name NOT LIKE '%Duel%' AND r.name NOT LIKE '%Clash%' AND r.name NOT LIKE '%All-Star%' AND r.name NOT LIKE '%All Star%'"
        " GROUP BY fs.driver_id"
        " HAVING COUNT(DISTINCT r.track_id) = ?",
        (year, segment, *track_ids, year, len(track_ids))
    ).fetchall()

    best = None
    for combo in itertools.combinations(rows, 4):
        total_sal = sum(c[2] for c in combo)
        if total_sal > 100:
            continue
        total_pts = sum(c[3] for c in combo)
        if best is None or total_pts > best["total_pts"]:
            best = {
                "drivers":      [{"name": c[1], "salary": c[2]} for c in combo],
                "total_salary": total_sal,
                "total_pts":    round(total_pts, 1),
                "_ids":         [c[0] for c in combo],
            }

    if best:
        ids = best["_ids"]
        conn.execute(
            "INSERT OR REPLACE INTO segment_optimal_lineups"
            " (year, segment, driver_1, driver_2, driver_3, driver_4, salary_total, segment_points)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (year, segment, ids[0], ids[1], ids[2], ids[3],
             best["total_salary"], best["total_pts"])
        )
        conn.commit()

    return best


def get_team_bonus_total(conn, team_name, track_ids, year):
    """
    Return total team bonus points earned by team_name across the segment
    identified by track_ids/year.  Returns 0 if no team_race_bonuses table
    or no data.
    """
    if not track_ids:
        return 0
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='team_race_bonuses'"
    ).fetchone():
        return 0
    ph  = ",".join("?" * len(track_ids))
    row = conn.execute(f"""
        SELECT COALESCE(SUM(trb.qual_bonus + trb.s1_bonus + trb.s2_bonus + trb.race_bonus), 0)
        FROM team_race_bonuses trb
        JOIN races r ON r.id = trb.race_id
        WHERE trb.team_name = ?
          AND r.track_id IN ({ph})
          AND r.year = ?
          AND r.name NOT LIKE '%Duel%'
          AND r.name NOT LIKE '%Clash%'
          AND r.name NOT LIKE '%All-Star%'
          AND r.name NOT LIKE '%All Star%'
    """, (team_name, *track_ids, year)).fetchone()
    return row[0] if row else 0


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table  = False
        self.in_cell   = False
        self.rows      = []
        self._cur_row  = []
        self._cell_txt = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        if self.in_table:
            if tag == "tr":
                self._cur_row = []
            if tag in ("td", "th"):
                self.in_cell   = True
                self._cell_txt = []

    def handle_endtag(self, tag):
        if not self.in_table:
            return
        if tag in ("td", "th"):
            self.in_cell = False
            self._cur_row.append(" ".join("".join(self._cell_txt).split()))
        if tag == "tr" and self._cur_row:
            self.rows.append(self._cur_row[:])
            self._cur_row = []

    def handle_data(self, data):
        if self.in_cell:
            self._cell_txt.append(data)


def scrape_standings(conn):
    row = conn.execute(
        "SELECT last_updated FROM league_standings_cache ORDER BY id LIMIT 1"
    ).fetchone()
    stale = True
    if row and row[0]:
        try:
            last  = datetime.datetime.fromisoformat(row[0])
            age   = (datetime.datetime.now() - last).total_seconds()
            stale = age > STANDINGS_TTL
        except Exception:
            stale = True

    if not stale:
        rows = conn.execute(
            "SELECT rank, name, points, behind, weekly, last_week"
            " FROM league_standings_cache ORDER BY id"
        ).fetchall()
        return [
            {"rank": r[0], "name": r[1], "points": r[2],
             "behind": r[3], "weekly": r[4], "last_week": r[5], "stale": False}
            for r in rows
        ]

    try:
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        resp = requests.get(STANDINGS_URL, headers=hdrs, timeout=15)
        resp.raise_for_status()
        parser = _TableParser()
        parser.feed(resp.text)

        standings = []
        for cells in parser.rows[1:]:
            if len(cells) < 2:
                continue
            standings.append({
                "rank":      cells[0] if len(cells) > 0 else "",
                "name":      cells[1] if len(cells) > 1 else "",
                "points":    cells[2] if len(cells) > 2 else "",
                "behind":    cells[3] if len(cells) > 3 else "",
                "weekly":    cells[4] if len(cells) > 4 else "",
                "last_week": cells[5] if len(cells) > 5 else "",
                "stale":     False,
            })

        if standings:
            now_str = datetime.datetime.now().isoformat()
            conn.execute("DELETE FROM league_standings_cache")
            for s in standings[:20]:
                conn.execute(
                    "INSERT INTO league_standings_cache"
                    " (rank, name, points, behind, weekly, last_week, last_updated)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (s["rank"], s["name"], s["points"],
                     s["behind"], s["weekly"], s["last_week"], now_str)
                )
            conn.commit()
            return standings[:20]

    except Exception as e:
        print("  [standings] scrape failed: " + str(e))

    cached_rows = conn.execute(
        "SELECT rank, name, points, behind, weekly, last_week, last_updated"
        " FROM league_standings_cache ORDER BY id"
    ).fetchall()
    if cached_rows:
        lu = cached_rows[0][6] if cached_rows[0][6] else "unknown"
        return [
            {"rank": r[0], "name": r[1], "points": r[2],
             "behind": r[3], "weekly": r[4], "last_week": r[5],
             "stale": True, "last_updated": lu}
            for r in cached_rows
        ]
    return []


def sal_badge(salary):
    return '<span class="sal">$' + str(salary) + "</span>"


def ppd_class(ppd):
    if ppd is None:
        return ""
    if ppd >= 10:
        return "v-high"
    if ppd >= 5:
        return "v-mid"
    return "v-low"


def table_html(rows, cols):
    if not rows:
        return '<p class="muted">No data available.</p>'
    html = ['<div class="table-wrap"><table><thead><tr>']
    for _, label in cols:
        html.append("<th>" + label + "</th>")
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for key, _ in cols:
            val  = row.get(key, "")
            cell = str(val) if val is not None else "&mdash;"
            if key == "salary":
                cell = sal_badge(val)
            elif key in ("pts_per_dollar", "ppd"):
                css  = ppd_class(val)
                cell = '<span class="' + css + '">' + str(val) + "</span>"
            elif key == "plus_minus":
                if val is None:
                    cell = "&mdash;"
                elif val > 0:
                    cell = '<span class="pm-pos">+' + str(val) + "</span>"
                elif val < 0:
                    cell = '<span class="pm-neg">' + str(val) + "</span>"
                else:
                    cell = "0"
            html.append("<td>" + cell + "</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


def track_tabs(track_histories):
    tab_btns   = []
    tab_panels = []
    cols = [
        ("driver", "Driver"), ("salary", "Salary"), ("starts", "Starts"),
        ("avg_pts", "Avg Pts"), ("pts_per_dollar", "Pts/$"),
        ("best_score", "Best"), ("avg_finish", "Avg Finish"),
    ]
    for i, (tname, rows) in enumerate(track_histories.items()):
        active = "active" if i == 0 else ""
        short  = tname.split()[0]
        tab_btns.append(
            '<button class="tab track-tab ' + active + '" onclick="showTrackTab(' + str(i) + ')">' + short + "</button>"
        )
        tab_panels.append(
            '<div class="tab-panel track-panel ' + active + '">'
            + '<p class="track-label">' + tname + "</p>"
            + table_html(rows, cols) + "</div>"
        )
    return (
        '<div class="tabs">' + "".join(tab_btns) + "</div>" +
        "".join(tab_panels)
    )


def race_results_tabs(race_results):
    cols = [
        ("driver",     "Driver"),
        ("finish_pos", "Finish"), ("salary", "Salary"),
        ("start_pos",  "Start"),  ("hist_avg", "Pre-'26 Avg"),
        ("plus_minus", "+/- Avg"),
    ]
    tab_btns   = []
    tab_panels = []
    for i, (tname, rows) in enumerate(race_results.items()):
        active = "active" if i == 0 else ""
        short  = tname.split()[0]
        tab_btns.append(
            '<button class="tab results-tab ' + active + '" onclick="showResultsTab(' + str(i) + ')">' + short + "</button>"
        )
        if rows:
            content = table_html(rows, cols)
        else:
            content = '<p class="race-pending">&#9873; Race not yet completed &mdash; check back after race day.</p>'
        tab_panels.append(
            '<div class="tab-panel results-panel ' + active + '">'
            + '<p class="track-label">' + tname + "</p>"
            + content + "</div>"
        )
    return (
        '<div class="tabs">' + "".join(tab_btns) + "</div>" +
        "".join(tab_panels)
    )


def _driver_chips(drivers):
    parts = []
    for d in drivers:
        parts.append(
            '<span class="driver-chip">' + d["name"] +
            ' <span class="driver-salary">$' + str(d["salary"]) + "</span></span>"
        )
    return "".join(parts)


def segment_intelligence_html(user, prev, optimal_prev, standings, cfg):
    yr, seg = cfg["year"], cfg["segment"]
    parts = [
        '<p style="font-size:0.75rem;color:var(--muted);margin-bottom:12px;">'
        '&#9432; Scores include qual, race &amp; stage pts. '
        'Stage pts require loading weekly PDFs via fetch_stages.py. '
        'Qualifying events (Duels, Clash, All-Star) excluded.</p>',
        '<div class="intel-grid">',
    ]

    # Panel 1: Current Segment Team
    parts.append('<div class="intel-card current intel-full">')
    parts.append('<div class="intel-label">Your Segment ' + str(seg) + " Team</div>")
    if user is None:
        parts.append(
            '<p class="intel-meta">No lineup saved yet. '
            "Run <code>python load_segment.py</code> and select your 4 drivers "
            "to start tracking your team.</p>"
        )
    else:
        parts.append('<div class="intel-chips">' + _driver_chips(user["drivers"]) + "</div>")
        bonus_line = ""
        if user.get("team_bonus", 0) > 0:
            bonus_line = (
                ' &nbsp;&bull;&nbsp; <span style="color:#2ecc71">+'
                + _fmt(user["team_bonus"]) + " team bonus</span>"
            )
        parts.append(
            '<div class="intel-pts">' + _fmt(user["total_pts"]) + "</div>"
            + '<div class="intel-meta">points through ' + str(user["races_done"])
            + " of " + str(user["total_races"]) + " races"
            + " &nbsp;&bull;&nbsp; $" + _fmt(user["total_salary"]) + " salary"
            + " &nbsp;&bull;&nbsp; $" + _fmt(100 - user["total_salary"]) + " remaining cap"
            + bonus_line + "</div>"
        )
    parts.append("</div>")

    # Panels 2 & 3 side by side
    parts.append('<div class="intel-two">')

    # Panel 2: Previous Segment
    parts.append('<div class="intel-card previous">')
    if prev is None:
        parts.append('<div class="intel-label">Previous Segment</div>')
        parts.append('<p class="intel-meta">No previous segment data yet.</p>')
    else:
        parts.append('<div class="intel-label">Segment ' + str(prev["segment"]) + " Your Team</div>")
        parts.append('<div class="intel-chips">' + _driver_chips(prev["drivers"]) + "</div>")
        bonus_meta = ""
        if prev.get("team_bonus", 0) > 0:
            bonus_meta = (
                ' &nbsp;&bull;&nbsp; <span style="color:#2ecc71">+'
                + _fmt(prev["team_bonus"]) + " team bonus</span>"
            )
        parts.append(
            '<div class="intel-pts">' + _fmt(prev["total_pts"]) + "</div>"
            + '<div class="intel-meta">points scored &nbsp;&bull;&nbsp; $'
            + _fmt(prev["total_salary"]) + " salary" + bonus_meta + "</div>"
        )
        if optimal_prev:
            gap = round(optimal_prev["total_pts"] - prev["total_pts"], 1)
            eff = round(prev["total_pts"] / optimal_prev["total_pts"] * 100, 1) if optimal_prev["total_pts"] > 0 else 0.0
            eff_cls = "eff-good" if eff >= 90 else ("eff-mid" if eff >= 70 else "eff-low")
            if gap > 0:
                table_html = ' &nbsp;&bull;&nbsp; left on table: <span style="color:var(--red)">-' + _fmt(gap) + ' pts</span>'
            elif gap < 0:
                table_html = ' &nbsp;&bull;&nbsp; <span style="color:#2ecc71">beat historical optimal by ' + _fmt(-gap) + ' pts</span>'
            else:
                table_html = ""
            parts.append(
                '<div class="intel-meta" style="margin-top:10px;">'
                + 'Efficiency: <span class="efficiency ' + eff_cls + '">' + _fmt(eff) + '%</span>'
                + table_html
                + '</div>'
            )
    parts.append("</div>")

    # Panel 3: Optimal Lineup
    parts.append('<div class="intel-card optimal">')
    if prev is None:
        parts.append('<div class="intel-label">Optimal Lineup</div>')
        parts.append('<p class="intel-meta">No previous segment data yet.</p>')
    elif optimal_prev is None:
        prev_seg = prev["segment"]
        parts.append('<div class="intel-label">Segment ' + str(prev_seg) + " Optimal</div>")
        parts.append('<p class="intel-meta">Optimal will appear once all segment races are complete.</p>')
    else:
        prev_seg = prev["segment"]
        parts.append('<div class="intel-label">Segment ' + str(prev_seg) + " Optimal</div>")
        parts.append('<div class="intel-chips">' + _driver_chips(optimal_prev["drivers"]) + "</div>")
        parts.append(
            '<div class="intel-pts">' + _fmt(optimal_prev["total_pts"]) + "</div>"
            + '<div class="intel-meta">max possible (no team bonus) &nbsp;&bull;&nbsp; $' + _fmt(optimal_prev["total_salary"]) + " salary</div>"
        )
        if prev:
            gap = round(optimal_prev["total_pts"] - prev["total_pts"], 1)
            if gap > 0:
                gap_html = 'Gap vs your team: <span style="color:#2ecc71">+' + _fmt(gap) + ' pts available</span>'
            else:
                gap_html = '<span style="color:#2ecc71">Your team beat the historical optimal by ' + _fmt(-gap) + ' pts</span>'
            parts.append(
                '<div class="intel-meta" style="margin-top:10px;">'
                + gap_html
                + '</div>'
            )
    parts.append("</div>")
    parts.append("</div>")  # end intel-two

    # Panel 4: League Standings
    parts.append('<div class="intel-card standings intel-full">')
    parts.append('<div class="intel-label">Fantasy League Standings</div>')

    if not standings:
        parts.append('<p class="intel-meta">Standings unavailable.</p>')
    else:
        stale = standings[0].get("stale", False)
        if stale:
            lu = standings[0].get("last_updated", "unknown")
            parts.append('<p class="intel-meta" style="color:#f39c12;">&#9888; Using cached standings from ' + lu + "</p>")
        else:
            parts.append('<p class="intel-meta" style="font-size:0.75rem;">Updated this session</p>')

        parts.append('<div class="table-wrap"><table><thead><tr>')
        for h in ["Pos", "Name", "Points", "Behind", "Weekly", "Last Week"]:
            parts.append("<th>" + h + "</th>")
        parts.append("</tr></thead><tbody>")
        for s in standings:
            is_me   = s.get("name", "").strip().lower() == USER_TEAM.lower()
            row_cls = ' class="my-team"' if is_me else ""
            parts.append("<tr" + row_cls + ">")
            for val in [s.get("rank",""), s.get("name",""), s.get("points",""),
                        s.get("behind",""), s.get("weekly",""), s.get("last_week","")]:
                parts.append("<td>" + str(val) + "</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    parts.append("</div>")  # end standings card
    parts.append("</div>")  # end intel-grid
    return "".join(parts)



# ── CSS / JS constants ─────────────────────────────────────────────────────────


# ── CSS / JS constants ─────────────────────────────────────────────────────────

_CSS = """
    :root {
      --bg:      #12121f;
      --surface: #1e1e35;
      --card:    #252540;
      --red:     #e63946;
      --yellow:  #f4d03f;
      --green:   #2ecc71;
      --text:    #eaeaea;
      --muted:   #7a7a9a;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.5; }

    /* ── Header ── */
    .checker {
      background: repeating-linear-gradient(
        45deg, #000 0, #000 12px, #fff 12px, #fff 24px
      );
      padding: 5px;
    }
    .header-inner {
      background: var(--red);
      padding: 28px 32px;
      text-align: center;
    }
    h1 { font-size: clamp(1.5rem, 5vw, 2.6rem); color: #fff;
          text-transform: uppercase; letter-spacing: 4px; }
    .sub { color: rgba(255,255,255,0.85); margin-top: 6px; font-size: 0.95rem; }
    .updated { color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 4px; }

    /* ── Layout ── */
    main { max-width: 1120px; margin: 0 auto; padding: 36px 16px; }
    section { margin-bottom: 52px; }
    h2 {
      color: var(--red); font-size: 0.8rem; text-transform: uppercase;
      letter-spacing: 3px; border-bottom: 2px solid var(--red);
      padding-bottom: 8px; margin-bottom: 20px;
    }

    /* ── Tables ── */
    .table-wrap { overflow-x: auto; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.855rem; }
    thead { position: sticky; top: 0; }
    th {
      background: var(--surface); color: var(--muted); padding: 10px 14px;
      text-align: left; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 1px; white-space: nowrap;
    }
    td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); white-space: nowrap; }
    tbody tr:hover td { background: rgba(255,255,255,0.025); }
    .sal {
      background: var(--red); color: #fff; border-radius: 4px;
      padding: 2px 8px; font-weight: 700; font-size: 0.8rem;
    }
    .v-high { color: #2ecc71; font-weight: 700; }
    .v-mid  { color: #f39c12; }
    .v-low  { color: var(--muted); }

    /* ── Tabs ── */
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .tab {
      background: var(--surface); border: none; color: var(--muted);
      padding: 8px 18px; border-radius: 6px; cursor: pointer;
      font-size: 0.85rem; transition: background 0.15s;
    }
    .tab:hover { background: var(--card); color: var(--text); }
    .tab.active { background: var(--red); color: #fff; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .track-label { color: var(--muted); font-size: 0.82rem; margin-bottom: 12px; }

    /* ── +/- avg coloring ── */
    .pm-pos  { color: #2ecc71; font-weight: 700; }
    .pm-neg  { color: #e74c3c; }
    .race-pending { color: var(--muted); font-style: italic; padding: 24px 0; }

    /* ── Driver chips ── */
    .driver-chip {
      display: inline-block; background: var(--surface); border-radius: 20px;
      padding: 4px 12px; margin: 3px 3px 0 0; font-size: 0.82rem;
    }
    .driver-salary { color: var(--yellow); font-weight: 700; }

    /* ── Segment Intelligence Grid ── */
    .intel-grid { display: grid; gap: 16px; margin-bottom: 0; }
    .intel-full { grid-column: 1 / -1; }

    .intel-two {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 640px) { .intel-two { grid-template-columns: 1fr; } }

    .intel-card {
      background: var(--card);
      border-radius: 10px;
      padding: 20px;
      border-left: 5px solid var(--muted);
    }
    .intel-card.current   { border-left-color: var(--yellow); }
    .intel-card.previous  { border-left-color: var(--red); }
    .intel-card.optimal   { border-left-color: #2ecc71; }
    .intel-card.standings { border-left-color: var(--muted); }

    .intel-label {
      font-size: 0.72rem; color: var(--muted);
      text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
    }
    .intel-pts  { font-size: 2rem; font-weight: 700; color: var(--yellow); }
    .intel-meta { font-size: 0.82rem; color: var(--muted); margin: 4px 0 14px; }
    .intel-chips { margin-bottom: 10px; }
    .efficiency { font-size: 1.1rem; font-weight: 700; }
    .eff-good { color: #2ecc71; }
    .eff-mid  { color: #f39c12; }
    .eff-low  { color: var(--red); }

    /* Standings table highlight */
    .my-team td { color: var(--yellow); font-weight: 700; }

    /* ── Footer ── */
    footer { text-align: center; color: var(--muted); font-size: 0.78rem; padding: 32px 16px; border-top: 1px solid rgba(255,255,255,0.05); }
    .muted { color: var(--muted); }
"""

_JS = """
  function showTrackTab(idx) {
    document.querySelectorAll('.track-tab').forEach(function(el, i) {
      el.classList.toggle('active', i === idx);
    });
    document.querySelectorAll('.track-panel').forEach(function(el, i) {
      el.classList.toggle('active', i === idx);
    });
  }
  function showResultsTab(idx) {
    document.querySelectorAll('.results-tab').forEach(function(el, i) {
      el.classList.toggle('active', i === idx);
    });
    document.querySelectorAll('.results-panel').forEach(function(el, i) {
      el.classList.toggle('active', i === idx);
    });
  }
"""


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(cfg, recent_form, track_histories, race_results,
               user_lineup, prev_lineup, optimal_prev, standings):
    yr      = cfg["year"]
    seg     = cfg["segment"]
    tnames  = cfg["track_names"]
    now     = datetime.datetime.now()
    updated = now.strftime("%B %d, %Y at %I:%M %p").replace(" 0", " ")

    tracks_str = " &bull; ".join(t.split()[0] for t in tnames)

    recent_cols = [
        ("driver",         "Driver"),
        ("salary",         "Salary"),
        ("avg_pts",        "Avg Pts"),
        ("pts_per_dollar", "Pts/$"),
        ("floor",          "Floor"),
        ("ceiling",        "Ceiling"),
    ]

    h = "<!DOCTYPE html>\n"
    h += '<html lang="en">\n<head>\n'
    h += '  <meta charset="utf-8">\n'
    h += '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
    h += "  <title>NASCAR Fantasy Picks &mdash; " + str(yr) + " Segment " + str(seg) + "</title>\n"
    h += "  <style>\n" + _CSS + "\n  </style>\n"
    h += "</head>\n<body>\n\n"

    # Header
    h += "<header>\n"
    h += '  <div class="checker">\n'
    h += '    <div class="header-inner">\n'
    h += '      <h1>&#127937; NASCAR Fantasy Picks</h1>\n'
    h += '      <p class="sub">' + str(yr) + " Segment " + str(seg) + " &nbsp;&bull;&nbsp; " + tracks_str + "</p>\n"
    h += '      <p class="updated">Updated ' + updated + "</p>\n"
    h += "    </div>\n  </div>\n</header>\n\n<main>\n\n"

    # Segment Intelligence
    h += "  <!-- Segment Intelligence -->\n"
    h += "  <section>\n"
    h += "    <h2>Segment Intelligence</h2>\n"
    h += "    " + segment_intelligence_html(user_lineup, prev_lineup, optimal_prev, standings, cfg) + "\n"
    h += "  </section>\n\n"

    # Recent Form
    h += "  <!-- Recent Form -->\n"
    h += "  <section>\n"
    h += "    <h2>Recent Form &mdash; Last 8 Completed Races</h2>\n"
    h += "    " + table_html(recent_form, recent_cols) + "\n"
    h += "  </section>\n\n"

    # Track History
    h += "  <!-- Track History -->\n"
    h += "  <section>\n"
    h += "    <h2>Track History</h2>\n"
    h += "    " + track_tabs(track_histories) + "\n"
    h += "  </section>\n\n"

    # Race Results
    h += "  <!-- Race Results -->\n"
    h += "  <section>\n"
    h += "    <h2>&#127937; Race Results &mdash; " + str(yr) + " Segment " + str(seg) + "</h2>\n"
    h += '    <p style="color:var(--muted);font-size:0.82rem;margin-bottom:16px;">\n'
    h += "      Finish &amp; start positions from this year&rsquo;s race.\n"
    h += "      Pre-26 Avg = avg finishing position at this track before 2026 (lower is better).\n"
    h += "      +/- Avg = spots better (green) or worse (red) than historical average.\n"
    h += "    </p>\n"
    h += "    " + race_results_tabs(race_results) + "\n"
    h += "  </section>\n\n"

    h += "</main>\n\n"
    h += "<footer>\n"
    h += "  NASCAR Fantasy Dashboard &bull; Data via ESPN API &bull; Braswell&rsquo;s Fantasy NASCAR League\n"
    h += "</footer>\n\n"
    h += "<script>\n" + _JS + "\n</script>\n\n"
    h += "</body>\n</html>\n"
    return h


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cfg  = load_config()
    conn = sqlite3.connect(DB_FILE)

    print("Building report for", cfg["year"], "Segment", cfg["segment"], "...")

    setup_segment_tables(conn)

    recent_form     = get_recent_form(conn, cfg)
    track_histories = get_track_histories(conn, cfg)
    race_results    = get_race_results(conn, cfg)

    user_lineup  = get_user_lineup(conn, cfg)
    prev_lineup  = get_prev_lineup(conn, cfg)

    prev_track_ids = None
    if prev_lineup:
        prev_track_ids = prev_lineup.get("track_ids")

    optimal_prev = None
    if prev_lineup and prev_track_ids:
        optimal_prev = get_or_compute_optimal(
            conn,
            cfg["year"],
            prev_lineup["segment"],
            prev_track_ids,
        )

    standings = scrape_standings(conn)

    conn.close()

    html = build_html(
        cfg, recent_form, track_histories, race_results,
        user_lineup, prev_lineup, optimal_prev, standings,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print("Done. Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
