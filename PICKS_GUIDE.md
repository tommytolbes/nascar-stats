# How to Pick Your Fantasy NASCAR Team
### Braswell's Fantasy NASCAR League — Step-by-Step Guide

---

## Overview

Each segment is 4 races. You pick 4 drivers under a $100 salary cap.
This guide walks you through everything, start to finish.

**Two situations:**
- **Start of a new segment** → do Steps 1 through 4
- **Already loaded the segment, just want to review picks** → skip to Step 3

---

## Step 1 — Open the Terminal

1. Press the **Windows key**, type **cmd**, and hit Enter
2. A black window will open (Command Prompt)
3. Type the following and press Enter:

```
cd "C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell"
```

You only need to do this once each time you open a new terminal window.

---

## Step 2 — Load the New Segment (do this once per segment)

Run this **after the last race of the previous segment:**

```
python load_segment.py
```

The script will ask you two questions:

**Question 1:** `Year (e.g. 2026):` → type the year and press Enter

**Question 2:** `Segment number (e.g. 2):` → type the segment number and press Enter

Next, it automatically pulls the latest driver salaries from Braswell's website.
You'll see a numbered list of every track in the database, like this:

```
  #    Track                                          Type
  --   -----                                          ----
  1    Atlanta Motor Speedway                         intermediate
  2    Bristol Motor Speedway                         short_track
  3    Charlotte Motor Speedway Road Course           road_course
  ...
```

**Question 3:** `Enter the 4 track numbers for this segment (space-separated):`

Look up which 4 tracks are in the upcoming segment on Braswell's website,
find their numbers in the list, and type them. Example: `1 2 15 20`

Confirm when it asks, and you're done. The system saves everything automatically.

---

## Step 3 — Run the Analysis

```
python query.py
```

This produces a long report. Scroll up after it finishes to read it from the top.
Here's what each section means:

---

### Section: Recent Form — Last 8 Completed Races

> Who has been running well lately, regardless of track type.

| Column | What it means |
|---|---|
| `salary` | Their cost this segment |
| `avg_fantasy_pts` | Average fantasy points over the last 8 races |
| `pts_per_dollar` | Value — higher is better (more points per $ spent) |
| `worst` | Their lowest single-race score recently |
| `best` | Their highest single-race score recently |

**How to use it:** A driver with a high average AND a decent floor (worst score)
is reliable. A high average with a very low floor is boom-or-bust.

---

### Section: Track History — [Each of the 4 Segment Tracks]

> How each driver has historically performed at that specific track.

You'll see one table per track. Same columns as Recent Form, plus:

| Column | What it means |
|---|---|
| `starts` | How many times they've raced at that track (in our data) |
| `avg_finish` | Average finishing position — lower is better (1st = best) |

**How to use it:** If a driver dominates one specific track type (superspeedway,
road course, short track), that history matters. Fewer starts means less
reliable data — treat 1–2 starts with some caution.

---

### Section: Segment X — Best Overall Value (All Tracks Combined)

> Each driver's average fantasy points AND cost efficiency across all 4 segment tracks.

| Column | What it means |
|---|---|
| `historical_starts` | Total starts at all 4 segment track types combined |
| `avg_fantasy_pts` | Average across those starts |
| `pts_per_dollar` | The key number — best bang for your buck |

**How to use it:** This is your primary ranking. Sort your attention here first.
Note that cheap drivers ($1–$4) often top this list because even average scores
look great per dollar. That's exactly why the optimizer pairs a few cheap drivers
with elite ones.

---

### Section: Driver Avg Fantasy Pts by Track Type

> Each driver's average, floor, and ceiling broken out by track category
> (superspeedway, intermediate, short_track, road_course).

**How to use it:** When a new segment has an unfamiliar track, look up that
track type here. Example: if Segment 3 has a road course you haven't studied,
find the top road course performers in this table.

---

### Section: Team Optimizer — Best 4-Driver Combos Under $100

> The computer's top 5 recommended teams, ranked by combined historical avg pts.

Each combo shows:
- Combined avg fantasy points
- Total salary and how much budget is left over
- The 4 drivers and their individual costs

**Example output:**
```
  #1  733.5 avg pts  |  $95 total  |  $5 leftover
       Joey Logano / Ryan Blaney / Chase Elliott / Austin Cindric
       $27 + $33 + $32 + $3
```

**How to use it:** The #1 combo is the strongest historically. But use your
own judgment too — if a driver is injured, in a slump, or switching teams,
override the computer. The optimizer looks backward; you look forward.

---

## Step 4 — Make Your Pick

Use this checklist to choose your 4 drivers:

- [ ] Start with the **Team Optimizer** — note the top 1–2 combos
- [ ] Cross-reference with **Recent Form** — is anyone in a recent slump?
- [ ] Check the **Track History** for each segment track — are there specialists?
- [ ] Check **Track Type** table if any track is a type you haven't seen before
- [ ] Make sure your 4 drivers total **$100 or under**
- [ ] Lock in your picks on Braswell's website before the first race of the segment

---

## Quick Reference — Segment Calendar

| When | What to do |
|---|---|
| After last race of old segment | Run `python load_segment.py` |
| Before first race of new segment | Run `python query.py`, make picks |
| Every Monday at noon | Database updates automatically (no action needed) |
| Anytime | Run `python query.py` to re-check your thinking |

---

## Troubleshooting

**"python is not recognized"**
→ Python isn't in your PATH. Try: `py load_segment.py` instead of `python load_segment.py`

**"No module named requests"**
→ Run: `pip install requests` and then try again

**"Could not reach salary page"**
→ Check your internet connection and try again. The website may be temporarily down.

**Recent Form shows no results**
→ The database may not have recent race data yet. Run `python fetch_races.py` to update.

**Optimizer shows no combinations**
→ The salary data for this segment may not be loaded yet. Run `python load_segment.py` first.
