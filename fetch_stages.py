"""
fetch_stages.py -- parse a Jayski race-result PDF and load stage results into nascar.db

Usage:
    python fetch_stages.py <pdf_path> <race_id>

Example:
    python fetch_stages.py "C:/Users/thoma/Downloads/12604_UNOFFRES.pdf" 202603080023

The script creates a stage_results table (if needed) and inserts/replaces
stage-finish positions for each driver in the given race.  Stage points and
bonuses are then computed by running build_fantasy.py.
"""

import sys
import sqlite3
import pdfplumber

DB_FILE = "nascar.db"

# x-coordinate column boundaries (from pdfplumber word extraction on Jayski PDFs)
X_CAR_MIN = 50    # car number column
X_CAR_MAX = 75
X_S1_MIN  = 355   # Stage 1 finish position (only top ~15 rows have a value here)
X_S1_MAX  = 398
X_S2_MIN  = 398   # Stage 2 finish position (only top ~15 rows have a value here)
X_S2_MAX  = 435

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS stage_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    TEXT    NOT NULL,
    driver_id  INTEGER NOT NULL,
    stage1_pos INTEGER,
    stage2_pos INTEGER,
    FOREIGN KEY (race_id)   REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id),
    UNIQUE (race_id, driver_id)
);
"""


def parse_pdf(pdf_path):
    car_stages = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)

            rows = {}
            for w in words:
                y_key = round(w["top"] / 3) * 3
                rows.setdefault(y_key, []).append(w)

            for _y, row_words in sorted(rows.items()):
                row_words.sort(key=lambda w: w["x0"])

                car_num = None
                s1_pos  = None
                s2_pos  = None

                for w in row_words:
                    x    = w["x0"]
                    text = w["text"].strip()

                    if X_CAR_MIN <= x <= X_CAR_MAX and text.isdigit():
                        car_num = text
                    elif X_S1_MIN <= x < X_S1_MAX and text.isdigit():
                        s1_pos = int(text)
                    elif X_S2_MIN <= x < X_S2_MAX and text.isdigit():
                        s2_pos = int(text)

                if car_num and (s1_pos is not None or s2_pos is not None):
                    entry = car_stages.setdefault(car_num, {"s1": None, "s2": None})
                    if s1_pos is not None:
                        entry["s1"] = s1_pos
                    if s2_pos is not None:
                        entry["s2"] = s2_pos

    return car_stages


def load_into_db(conn, race_id, car_stages):
    car_to_driver = {}
    for car_num, driver_id in conn.execute(
        "SELECT car_number, driver_id FROM race_results WHERE race_id = ?", (race_id,)
    ).fetchall():
        if car_num:
            car_to_driver[str(car_num)] = driver_id

    inserted = 0
    skipped  = []

    for car_num, stages in car_stages.items():
        driver_id = car_to_driver.get(car_num)
        if driver_id is None:
            skipped.append(car_num)
            continue

        conn.execute("""
            INSERT OR REPLACE INTO stage_results
                (race_id, driver_id, stage1_pos, stage2_pos)
            VALUES (?, ?, ?, ?)
        """, (race_id, driver_id, stages["s1"], stages["s2"]))
        inserted += 1

    conn.commit()
    print(f"Inserted/replaced {inserted} stage result rows.")
    if skipped:
        print(f"  Could not match car numbers to drivers: {skipped}")

    return inserted


def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_stages.py <pdf_path> <race_id>")
        print("  race_id examples (from nascar.db races table):")
        print("    Daytona 500 2026 : 202602150001")
        print("    Atlanta 2026     : 202602220025")
        print("    COTA 2026        : 202603013998")
        print("    Phoenix 2026     : 202603080023")
        sys.exit(1)

    pdf_path = sys.argv[1]
    race_id  = sys.argv[2]

    conn = sqlite3.connect(DB_FILE)
    conn.execute(SETUP_SQL)
    conn.commit()

    race = conn.execute(
        "SELECT name, date FROM races WHERE id = ?", (race_id,)
    ).fetchone()
    if not race:
        print(f"ERROR: race_id {race_id!r} not found in nascar.db")
        conn.close()
        sys.exit(1)

    print(f"Race  : {race[0]} ({race[1]})")
    print(f"PDF   : {pdf_path}")
    print()

    car_stages = parse_pdf(pdf_path)

    if not car_stages:
        print("WARNING: no stage data found in PDF -- check column x-bounds.")
        conn.close()
        sys.exit(1)

    print(f"Found stage data for {len(car_stages)} driver(s):")
    sorted_cars = sorted(
        car_stages.items(),
        key=lambda x: (x[1]["s1"] if x[1]["s1"] is not None else 99,
                        x[1]["s2"] if x[1]["s2"] is not None else 99)
    )
    for car_num, stages in sorted_cars:
        s1 = stages["s1"] if stages["s1"] is not None else "--"
        s2 = stages["s2"] if stages["s2"] is not None else "--"
        print(f"  Car #{car_num:>3}  Stage1={s1}  Stage2={s2}")

    print()
    load_into_db(conn, race_id, car_stages)
    conn.close()
    print()
    print("Done. Run python build_fantasy.py to update stage points in fantasy_scores.")


if __name__ == "__main__":
    main()
