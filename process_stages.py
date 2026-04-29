"""
process_stages.py -- Auto-process new Jayski PDF stage result files.

Scans the project directory (and an optional pdfs/ staging folder) for
*_UNOFFRES.pdf files, matches each to a race in nascar.db by date, and
loads stage results using fetch_stages logic. Skips PDFs whose race already
has stage_results loaded.

Usage:
    python process_stages.py        # process all new PDFs found

Called automatically by update.bat between fetch_races.py and build_fantasy.py.
To add a new race's stage data, drop the *_UNOFFRES.pdf into the project
directory or the pdfs/ subfolder and re-run update.bat.
"""

import os
import re
import glob
import sqlite3
import datetime
import pdfplumber

import fetch_stages  # reuse parse_pdf, load_into_db, SETUP_SQL

DB_FILE = "nascar.db"

# Directories to scan for *_UNOFFRES.pdf files
PDF_DIRS = [
    ".",      # project root
    "pdfs",   # optional staging folder
]


def extract_race_date(pdf_path):
    """
    Parse the race date from a Jayski PDF header line.
    Looks for: "Race Results for the X - [Day], [Month DD, YYYY]"
    Returns a datetime.date, or None if not found.
    """
    date_pattern = re.compile(
        r"Race Results for .+?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
        r"(\w+ \d{1,2},\s+\d{4})",
        re.IGNORECASE,
    )
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text() or ""
        m = date_pattern.search(text)
        if m:
            date_str = m.group(1).strip()
            return datetime.datetime.strptime(date_str, "%B %d, %Y").date()
    except Exception as e:
        print(f"  [warn] Could not read {pdf_path}: {e}")
    return None


def find_race_id(conn, race_date):
    """Look up a race_id by date. Returns (race_id, race_name) or (None, None)."""
    row = conn.execute(
        "SELECT id, name FROM races WHERE date = ? "
        "AND name NOT LIKE '%Duel%' "
        "AND name NOT LIKE '%Clash%' "
        "AND name NOT LIKE '%All-Star%' "
        "AND name NOT LIKE '%All Star%'",
        (str(race_date),),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def stage_results_exist(conn, race_id):
    """Return True if stage_results rows already exist for this race."""
    row = conn.execute(
        "SELECT COUNT(*) FROM stage_results WHERE race_id = ?", (race_id,)
    ).fetchone()
    return (row[0] > 0) if row else False


def find_pdfs():
    """Find all *_UNOFFRES.pdf files across configured directories."""
    found = []
    for d in PDF_DIRS:
        if os.path.isdir(d):
            found.extend(glob.glob(os.path.join(d, "*_UNOFFRES.pdf")))
    seen, result = set(), []
    for p in found:
        norm = os.path.normpath(p)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return sorted(result)


def main():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(fetch_stages.SETUP_SQL)
    conn.commit()

    pdfs = find_pdfs()
    if not pdfs:
        print("process_stages: no *_UNOFFRES.pdf files found.")
        conn.close()
        return

    print(f"process_stages: found {len(pdfs)} PDF(s) to check.")

    loaded  = 0
    skipped = 0

    for pdf_path in pdfs:
        basename = os.path.basename(pdf_path)
        print(f"\n  [{basename}]")

        race_date = extract_race_date(pdf_path)
        if race_date is None:
            print(f"    Could not parse date — skipping.")
            skipped += 1
            continue

        race_id, race_name = find_race_id(conn, race_date)
        if race_id is None:
            print(f"    No race in DB for {race_date} — skipping.")
            skipped += 1
            continue

        if stage_results_exist(conn, race_id):
            print(f"    {race_name} ({race_date}) — already loaded, skipping.")
            skipped += 1
            continue

        print(f"    {race_name} ({race_date}) — loading stage data...")
        car_stages = fetch_stages.parse_pdf(pdf_path)
        if not car_stages:
            print(f"    WARNING: no stage data found in PDF — check x-bounds in fetch_stages.py.")
            skipped += 1
            continue

        fetch_stages.load_into_db(conn, race_id, car_stages)
        loaded += 1

    conn.close()
    print(f"\nprocess_stages: {loaded} race(s) loaded, {skipped} skipped.")


if __name__ == "__main__":
    main()
