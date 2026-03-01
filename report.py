"""
NASCAR Fantasy Report Generator
---------------------------------
Generates index.html — a webpage version of your fantasy analysis.

Reads from nascar.db and segment.json, then writes index.html.
Commit and push index.html to GitHub Pages to publish it online.

Usage:   python report.py
Publish: git add index.html && git commit -m "update" && git push
"""

import sqlite3
import json
import os
import itertools
import datetime

DB_FILE     = "nascar.db"
CONFIG_FILE = "segment.json"
OUTPUT_FILE = "index.html"


# ── Data helpers ───────────────────────────────────────────────────────────────

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


def get_optimizer(conn, cfg):
    yr, seg, tids = cfg["year"], cfg["segment"], cfg["track_ids"]
    ph = ",".join("?" * len(tids))
    rows = conn.execute(f"""
        WITH seg AS (
            SELECT r.id AS race_id FROM races r
            JOIN tracks t ON t.id = r.track_id WHERE t.id IN ({ph})
        )
        SELECT d.display_name AS name, ds.salary,
               ROUND(AVG(fs.total_pts), 1) AS avg_pts
        FROM fantasy_scores fs
        JOIN seg ON seg.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id HAVING COUNT(*) >= 2
        ORDER BY avg_pts DESC
    """, (*tids, yr, seg)).fetchall()

    combos = []
    for combo in itertools.combinations(rows, 4):
        total_sal = sum(c[1] for c in combo)
        if total_sal > 100:
            continue
        total_pts = sum(c[2] for c in combo)
        combos.append((round(total_pts, 1), total_sal, combo))
    combos.sort(reverse=True)
    return combos[:5]


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


def get_overall_value(conn, cfg):
    yr, seg, tids = cfg["year"], cfg["segment"], cfg["track_ids"]
    ph = ",".join("?" * len(tids))
    return q(conn, f"""
        WITH seg AS (
            SELECT r.id AS race_id FROM races r
            JOIN tracks t ON t.id = r.track_id WHERE t.id IN ({ph})
        )
        SELECT d.display_name AS driver, ds.salary,
               COUNT(*) AS starts,
               ROUND(AVG(fs.total_pts), 1)             AS avg_pts,
               ROUND(AVG(fs.total_pts) / ds.salary, 2) AS pts_per_dollar
        FROM fantasy_scores fs
        JOIN seg ON seg.race_id = fs.race_id
        JOIN drivers d ON d.id = fs.driver_id
        JOIN driver_salaries ds ON ds.driver_id = fs.driver_id
            AND ds.year = ? AND ds.segment = ?
        GROUP BY fs.driver_id HAVING starts >= 2
        ORDER BY pts_per_dollar DESC LIMIT 20
    """, (*tids, yr, seg))


# ── HTML helpers ───────────────────────────────────────────────────────────────

def sal_badge(salary):
    return f'<span class="sal">${salary}</span>'


def ppd_class(ppd):
    if ppd is None:
        return ""
    if ppd >= 10:
        return "v-high"
    if ppd >= 5:
        return "v-mid"
    return "v-low"


def table_html(rows, cols):
    """Render a list of dicts as an HTML table. cols = [(key, label), ...]"""
    if not rows:
        return '<p class="muted">No data available.</p>'
    html = ['<div class="table-wrap"><table><thead><tr>']
    for _, label in cols:
        html.append(f"<th>{label}</th>")
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for key, _ in cols:
            val = row.get(key, "")
            cell = str(val) if val is not None else "—"
            css = ""
            if key == "salary":
                cell = sal_badge(val)
            elif key in ("pts_per_dollar", "ppd"):
                css = ppd_class(val)
                cell = f'<span class="{css}">{val}</span>'
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


def optimizer_cards(combos):
    border = ["gold", "silver", "bronze", "", ""]
    medals = ["🥇", "🥈", "🥉", "4", "5"]
    html   = ['<div class="combos">']
    for i, (pts, salary, combo) in enumerate(combos):
        cls  = f"combo-card {border[i]}" if border[i] else "combo-card"
        drivers_html = "".join(
            f'<span class="driver-chip">{c[0]} <span class="driver-salary">${c[1]}</span></span>'
            for c in combo
        )
        costs = " + ".join(f"${c[1]}" for c in combo)
        html.append(f"""
        <div class="{cls}">
          <div class="combo-rank">{medals[i]}  Combo #{i+1}</div>
          <div class="combo-pts">{pts} <span style="font-size:1rem;color:var(--muted)">avg pts</span></div>
          <div class="combo-cost">${salary} total &bull; ${100-salary} leftover &bull; {costs}</div>
          <div class="combo-drivers">{drivers_html}</div>
        </div>""")
    html.append("</div>")
    return "".join(html)


def track_tabs(track_histories):
    """Render tabbed track history sections."""
    tab_btns  = []
    tab_panels = []
    cols = [
        ("driver", "Driver"), ("salary", "Salary"), ("starts", "Starts"),
        ("avg_pts", "Avg Pts"), ("pts_per_dollar", "Pts/$"),
        ("best_score", "Best"), ("avg_finish", "Avg Finish"),
    ]
    for i, (tname, rows) in enumerate(track_histories.items()):
        active = "active" if i == 0 else ""
        short  = tname.split()[0]  # first word of track name
        tab_btns.append(
            f'<button class="tab {active}" onclick="showTab(\'track\',{i})" data-group="track">{short}</button>'
        )
        tab_panels.append(
            f'<div class="tab-panel {active}" data-group="track">'
            f'<p class="track-label">{tname}</p>'
            f'{table_html(rows, cols)}</div>'
        )
    return (
        '<div class="tabs">' + "".join(tab_btns) + "</div>" +
        "".join(tab_panels)
    )


# ── Main HTML builder ──────────────────────────────────────────────────────────

def build_html(cfg, optimizer, recent_form, track_histories, overall_value):
    yr      = cfg["year"]
    seg     = cfg["segment"]
    tnames  = cfg["track_names"]
    now     = datetime.datetime.now()
    updated = now.strftime("%B %d, %Y at %I:%M %p").replace(" 0", " ")

    tracks_str = " &bull; ".join(t.split()[0] for t in tnames)

    recent_cols = [
        ("driver", "Driver"), ("salary", "Salary"),
        ("avg_pts", "Avg Pts"), ("pts_per_dollar", "Pts/$"),
        ("floor", "Floor"), ("ceiling", "Ceiling"),
    ]
    value_cols = [
        ("driver", "Driver"), ("salary", "Salary"),
        ("starts", "Starts"), ("avg_pts", "Avg Pts"), ("pts_per_dollar", "Pts/$"),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NASCAR Fantasy Picks &mdash; {yr} Segment {seg}</title>
  <style>
    :root {{
      --bg:      #12121f;
      --surface: #1e1e35;
      --card:    #252540;
      --red:     #e63946;
      --yellow:  #f4d03f;
      --green:   #2ecc71;
      --text:    #eaeaea;
      --muted:   #7a7a9a;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.5; }}

    /* ── Header ── */
    .checker {{
      background: repeating-linear-gradient(
        45deg, #000 0, #000 12px, #fff 12px, #fff 24px
      );
      padding: 5px;
    }}
    .header-inner {{
      background: var(--red);
      padding: 28px 32px;
      text-align: center;
    }}
    h1 {{ font-size: clamp(1.5rem, 5vw, 2.6rem); color: #fff;
          text-transform: uppercase; letter-spacing: 4px; }}
    .sub {{ color: rgba(255,255,255,0.85); margin-top: 6px; font-size: 0.95rem; }}
    .updated {{ color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 4px; }}

    /* ── Layout ── */
    main {{ max-width: 1120px; margin: 0 auto; padding: 36px 16px; }}
    section {{ margin-bottom: 52px; }}
    h2 {{
      color: var(--red); font-size: 0.8rem; text-transform: uppercase;
      letter-spacing: 3px; border-bottom: 2px solid var(--red);
      padding-bottom: 8px; margin-bottom: 20px;
    }}

    /* ── Optimizer cards ── */
    .combos {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
    .combo-card {{
      background: var(--card); border-radius: 10px; padding: 20px;
      border-left: 5px solid var(--muted);
    }}
    .combo-card.gold   {{ border-left-color: #f4d03f; }}
    .combo-card.silver {{ border-left-color: #bbb; }}
    .combo-card.bronze {{ border-left-color: #cd7f32; }}
    .combo-rank {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
    .combo-pts  {{ font-size: 1.9rem; font-weight: 700; color: var(--yellow); line-height: 1.1; }}
    .combo-cost {{ font-size: 0.82rem; color: var(--muted); margin: 6px 0 14px; }}
    .driver-chip {{
      display: inline-block; background: var(--surface); border-radius: 20px;
      padding: 4px 12px; margin: 3px 3px 0 0; font-size: 0.82rem;
    }}
    .driver-salary {{ color: var(--yellow); font-weight: 700; }}

    /* ── Tables ── */
    .table-wrap {{ overflow-x: auto; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.855rem; }}
    thead {{ position: sticky; top: 0; }}
    th {{
      background: var(--surface); color: var(--muted); padding: 10px 14px;
      text-align: left; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 1px; white-space: nowrap;
    }}
    td {{ padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); white-space: nowrap; }}
    tbody tr:hover td {{ background: rgba(255,255,255,0.025); }}
    .sal {{
      background: var(--red); color: #fff; border-radius: 4px;
      padding: 2px 8px; font-weight: 700; font-size: 0.8rem;
    }}
    .v-high {{ color: #2ecc71; font-weight: 700; }}
    .v-mid  {{ color: #f39c12; }}
    .v-low  {{ color: var(--muted); }}

    /* ── Tabs ── */
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    .tab {{
      background: var(--surface); border: none; color: var(--muted);
      padding: 8px 18px; border-radius: 6px; cursor: pointer;
      font-size: 0.85rem; transition: background 0.15s;
    }}
    .tab:hover {{ background: var(--card); color: var(--text); }}
    .tab.active {{ background: var(--red); color: #fff; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .track-label {{ color: var(--muted); font-size: 0.82rem; margin-bottom: 12px; }}

    /* ── Footer ── */
    footer {{ text-align: center; color: var(--muted); font-size: 0.78rem; padding: 32px 16px; border-top: 1px solid rgba(255,255,255,0.05); }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>

<header>
  <div class="checker">
    <div class="header-inner">
      <h1>&#127937; NASCAR Fantasy Picks</h1>
      <p class="sub">{yr} Segment {seg} &nbsp;&bull;&nbsp; {tracks_str}</p>
      <p class="updated">Updated {updated}</p>
    </div>
  </div>
</header>

<main>

  <!-- ── Optimizer ── -->
  <section>
    <h2>&#127942; Top Team Combinations</h2>
    {optimizer_cards(optimizer)}
  </section>

  <!-- ── Recent Form ── -->
  <section>
    <h2>Recent Form &mdash; Last 8 Completed Races</h2>
    {table_html(recent_form, recent_cols)}
  </section>

  <!-- ── Track History ── -->
  <section>
    <h2>Track History</h2>
    {track_tabs(track_histories)}
  </section>

  <!-- ── Overall Value ── -->
  <section>
    <h2>Best Overall Value &mdash; All Segment Tracks Combined</h2>
    {table_html(overall_value, value_cols)}
  </section>

</main>

<footer>
  NASCAR Fantasy Dashboard &bull; Data via ESPN API &bull; Braswell&rsquo;s Fantasy NASCAR League
</footer>

<script>
  function showTab(group, idx) {{
    document.querySelectorAll('[data-group="' + group + '"]').forEach(function(el, i) {{
      el.classList.toggle('active', i === idx);
    }});
  }}
</script>

</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cfg  = load_config()
    conn = sqlite3.connect(DB_FILE)

    print(f"Building report for {cfg['year']} Segment {cfg['segment']}...")

    optimizer       = get_optimizer(conn, cfg)
    recent_form     = get_recent_form(conn, cfg)
    track_histories = get_track_histories(conn, cfg)
    overall_value   = get_overall_value(conn, cfg)

    conn.close()

    html = build_html(cfg, optimizer, recent_form, track_histories, overall_value)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Done. Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
