"""
NASCAR Segment Loader
---------------------
Run this at the start of each new fantasy segment (every 4 weeks).

It will:
  1. Scrape the current driver salaries from Braswell's website
  2. Show you all tracks -- you pick the 4 for this segment
  3. Save salaries + track config to the database and segment.json

After running this, 'python query.py' will automatically reflect
the new segment.

Usage:   python load_segment.py
"""

import requests
import sqlite3
import json
import re
import os
from html.parser import HTMLParser
from difflib import SequenceMatcher

DB_FILE    = "nascar.db"
CONFIG_FILE = "segment.json"
SALARY_URL = "https://www.braswellsfantasynascar.com/teamselection.html"


# ── HTML text extractor ────────────────────────────────────────────────────────

class TextExtractor(HTMLParser):
    """Strips HTML tags and returns plain text."""
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag in ("br", "p", "li", "tr", "div", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


# ── Salary scraper ─────────────────────────────────────────────────────────────

def scrape_salaries():
    """Fetch driver salaries from Braswell's team selection page."""
    print(f"\nFetching salaries from Braswell's website...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
    try:
        r = requests.get(SALARY_URL, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR: Could not reach salary page: {e}")
        return []

    parser = TextExtractor()
    parser.feed(r.text)
    text = parser.get_text()

    # Match patterns like "Kyle Larson, $40" or "Kyle Larson $40"
    pattern = re.compile(r"([A-Z][a-zA-Z]+(?:[\s\.\-'][a-zA-Z]+)+)\s*[,\-]?\s*\$(\d+)")
    seen    = set()
    results = []

    for match in pattern.finditer(text):
        name  = " ".join(match.group(1).split())  # normalize whitespace
        price = int(match.group(2))
        if 1 <= price <= 60 and name not in seen:
            seen.add(name)
            results.append((name, price))

    return results


# ── Driver name matcher ────────────────────────────────────────────────────────

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_driver(conn, scraped_name, all_drivers):
    """
    Find the best matching driver in the DB for a scraped name.
    Returns (driver_id, display_name, score) or None.
    """
    best_id, best_name, best_score = None, None, 0.0

    for driver_id, display_name in all_drivers:
        score = similarity(scraped_name, display_name)
        if score > best_score:
            best_score = score
            best_id    = driver_id
            best_name  = display_name

    if best_score >= 0.78:
        return best_id, best_name, best_score
    return None


# ── Track picker ───────────────────────────────────────────────────────────────

def pick_tracks(conn):
    """Show all tracks grouped by type; user picks 4 by number."""
    tracks = conn.execute(
        "SELECT id, full_name, track_type FROM tracks ORDER BY track_type, full_name"
    ).fetchall()

    print("\nAvailable tracks:")
    print(f"  {'#':<4}  {'Track':<45}  Type")
    print(f"  {'--':<4}  {'-----':<45}  ----")
    for i, (tid, name, ttype) in enumerate(tracks, 1):
        print(f"  {i:<4}  {name:<45}  {ttype}")

    print()
    while True:
        raw = input("Enter the 4 track numbers for this segment (space-separated): ").strip()
        parts = raw.split()
        if len(parts) != 4:
            print("  Please enter exactly 4 numbers.")
            continue
        try:
            indices  = [int(n) - 1 for n in parts]
            selected = [tracks[i] for i in indices]
            break
        except (ValueError, IndexError):
            print("  Invalid selection. Try again.")

    print("\nSelected tracks:")
    for tid, name, ttype in selected:
        print(f"  - {name}  ({ttype})")

    confirm = input("\nLook right? (y/n): ").strip().lower()
    if confirm != "y":
        return pick_tracks(conn)

    return [(tid, name) for tid, name, _ in selected]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  NASCAR Segment Loader")
    print("=" * 55)

    conn = sqlite3.connect(DB_FILE)

    # ── Step 1: Segment info ───────────────────────────────────────────────────
    print()
    year    = int(input("Year (e.g. 2026): ").strip())
    segment = int(input("Segment number (e.g. 2): ").strip())

    # ── Step 2: Scrape salaries ────────────────────────────────────────────────
    raw = scrape_salaries()
    if not raw:
        print("\n  Could not load salaries from the website.")
        print("  Check your internet connection and try again.")
        conn.close()
        return

    print(f"  Found {len(raw)} drivers on the salary page.")

    # Load all DB drivers for matching
    all_drivers = conn.execute(
        "SELECT id, display_name FROM drivers"
    ).fetchall()

    matched   = []
    unmatched = []

    for scraped_name, salary in raw:
        result = match_driver(conn, scraped_name, all_drivers)
        if result:
            driver_id, db_name, score = result
            matched.append((driver_id, salary, scraped_name, db_name))
        else:
            unmatched.append((scraped_name, salary))

    print(f"  Matched {len(matched)} of {len(raw)} drivers to the database.")

    if unmatched:
        print(f"\n  Could not match {len(unmatched)} drivers (will be skipped):")
        for name, salary in unmatched:
            print(f"    ${salary:>2}  {name}")
        print()

    # ── Step 3: Pick tracks ────────────────────────────────────────────────────
    track_pairs = pick_tracks(conn)   # [(id, name), ...]

    # ── Step 4: Save to database ───────────────────────────────────────────────
    # Salaries
    conn.execute(
        "DELETE FROM driver_salaries WHERE year = ? AND segment = ?",
        (year, segment)
    )
    for driver_id, salary, _, _ in matched:
        conn.execute(
            "INSERT OR REPLACE INTO driver_salaries (driver_id, year, segment, salary) VALUES (?,?,?,?)",
            (driver_id, year, segment, salary)
        )

    # Segments table
    conn.execute(
        "DELETE FROM segments WHERE year = ? AND segment = ?",
        (year, segment)
    )
    for tid, tname in track_pairs:
        conn.execute(
            "INSERT INTO segments (segment, year, race_name, slug) VALUES (?,?,?,?)",
            (segment, year, tname, tname.lower().replace(" ", "_"))
        )

    conn.commit()
    conn.close()

    # ── Step 5: Write segment.json ─────────────────────────────────────────────
    config = {
        "year":        year,
        "segment":     segment,
        "track_ids":   [t[0] for t in track_pairs],
        "track_names": [t[1] for t in track_pairs],
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Segment {segment} ({year}) loaded successfully!")
    print(f"  {len(matched)} driver salaries saved.")
    print(f"  Tracks: {', '.join(t[1] for t in track_pairs)}")
    print(f"\n  Run 'python query.py' to see your picks.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
