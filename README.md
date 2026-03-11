# Braswell's Fantasy NASCAR — Stats & Analysis Tool

A local Python tool that tracks historical NASCAR Cup Series data, calculates fantasy scores, and generates a live analysis website for the Braswell's Fantasy NASCAR league.

---

## What It Does

- Pulls NASCAR Cup Series race results from ESPN's public API (2000–present)
- Calculates fantasy points per driver per race using the Braswell league scoring system
- Tracks all 42 league teams' driver picks and computes team bonuses each race
- Generates a static HTML dashboard (`index.html`) with driver analysis, track history, optimizer combos, and a live Segment Intelligence panel
- Auto-updates weekly via Windows Task Scheduler and pushes to GitHub Pages

---

## Scoring System

**Individual driver points (per race):**
| Category | Points |
|---|---|
| Race finish position | 300 (1st) down to 5 (41st) |
| Qualifying position | 75 (1st) down to 1 (15th) |
| Stage finish position | Same scale as qualifying |

**Team bonuses (per race, awarded to 1 team):**
| Category | Bonus |
|---|---|
| Highest combined qualifying pts | +25 |
| Highest combined Stage 1 pts | +25 |
| Highest combined Stage 2 pts | +25 |
| Highest combined race pts | +100 |

Each team picks 4 drivers under a **$100 salary cap** per segment. Segments are 4 races long.

---

## Scripts

| Script | What it does | When to run |
|---|---|---|
| `main.py` | Pulls historical standings from ESPN API | Once to build the DB |
| `fetch_races.py` | Pulls race-by-race results from ESPN API | Weekly (auto via update.bat) |
| `fetch_stages.py` | Parses a Jayski PDF to load stage results | After each race |
| `build_fantasy.py` | Calculates fantasy scores and team bonuses | After fetch_races / fetch_stages |
| `load_segment.py` | Loads a new segment (salaries, tracks, your lineup) | Start of each new segment |
| `fetch_teams.py` | Loads all 42 league teams' driver picks | Once per segment |
| `report.py` | Generates index.html | Weekly (auto via update.bat) |
| `query.py` | Prints analysis to terminal | Anytime |
| `update.bat` | Runs the full weekly update pipeline | Auto via Task Scheduler |

---

## Setup

**Requirements:** Python 3.x, plus:
```
pip install requests pdfplumber
```

**First-time database build:**
```
python main.py
python fetch_races.py
python build_fantasy.py
```

**Start of a new segment:**
```
python load_segment.py       # enter year, segment, tracks, and your lineup
python fetch_teams.py        # load all 42 league teams' picks
python build_fantasy.py      # compute fantasy scores and team bonuses
python report.py             # regenerate the website
```

**After each race:**
```
python fetch_stages.py <pdf_path> <race_id>   # load stage results from Jayski PDF
python build_fantasy.py                        # recalculate scores
python report.py                               # regenerate the website
```

---

## Database Tables

| Table | Contents |
|---|---|
| `drivers` | Driver names and IDs |
| `tracks` | Track names, types, locations |
| `races` | One row per race |
| `race_results` | Finish position, start position, laps led per driver per race |
| `stage_results` | Stage 1 and Stage 2 finish positions per driver per race |
| `driver_salaries` | Driver salary per year/segment |
| `fantasy_scores` | Calculated fantasy points per driver per race |
| `league_team_picks` | All 42 teams' 4 driver picks per segment |
| `team_race_bonuses` | Team bonus points awarded per race |
| `segment_lineups` | Your 4-driver lineup per segment |
| `segment_optimal_lineups` | Best possible lineup (computed after segment ends) |

---

## Website

The generated `index.html` (hosted on GitHub Pages) includes:

- **Recent Form** — last 8 races avg fantasy pts and pts/dollar
- **Track History** — historical performance at each segment track
- **Track Type Breakdown** — avg pts by superspeedway / intermediate / short track / road course
- **Team Optimizer** — top 5 driver combos under $100
- **Segment Intelligence** — your current lineup vs. optimal, plus league standings

See `PICKS_GUIDE.md` for a step-by-step guide to using the tool each segment.

---

## Automation

`update.bat` is scheduled via Windows Task Scheduler to run every Monday at noon. It:
1. Pulls the latest race results (`fetch_races.py`)
2. Recalculates fantasy scores (`build_fantasy.py`)
3. Regenerates the website (`report.py`)
4. Commits and pushes to GitHub Pages
