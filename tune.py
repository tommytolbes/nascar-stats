"""
NASCAR Model Hyperparameter Tuner
------------------------------------
Grid search over H, phi, K to minimize MAE on historical race predictions.
Uses time-series leave-one-out cross-validation.

Usage:   python tune.py

Writes:
  params.json                     — active parameters (overwrites)
  params_YYYYMMDD_HHMM.json      — timestamped archive

Runtime: ~5-15 minutes depending on dataset size.
"""

import sqlite3
import json
import itertools
import numpy as np
from datetime import datetime

import model

DB_FILE     = "nascar.db"
PARAMS_FILE = "params.json"

# Grid search space
H_VALUES   = [4, 6, 8, 10, 12, 16, 20]
PHI_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
K_VALUES   = [5, 8, 10, 15]


def load_existing_params(path=PARAMS_FILE):
    """Load existing params.json to preserve non-tuned keys."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def fetch_all_races_ordered(conn):
    """Return all race IDs in chronological order."""
    return [
        row[0] for row in conn.execute(
            "SELECT id FROM races ORDER BY date ASC"
        ).fetchall()
    ]


def evaluate_params(conn, H, phi, K, all_race_ids, yr_current=2026):
    """
    Leave-one-out time-series CV: for each race (skip first 10 as warm-up),
    predict each driver's score using all prior races, compute MAE.

    Returns mean absolute error across all predictions.
    """
    errors = []
    WARMUP = 10  # skip first N races — not enough history

    for target_idx in range(WARMUP, len(all_race_ids)):
        target_race_id = all_race_ids[target_idx]

        # Get drivers and actual scores for target race
        actuals = conn.execute("""
            SELECT fs.driver_id, fs.total_pts
            FROM fantasy_scores fs
            WHERE fs.race_id = ?
        """, (target_race_id,)).fetchall()

        if not actuals:
            continue

        # Get target race year for N computation
        target_year = conn.execute(
            "SELECT year FROM races WHERE id = ?", (target_race_id,)
        ).fetchone()
        if not target_year:
            continue
        target_year = target_year[0]

        # Get target track type
        track_type = conn.execute("""
            SELECT t.track_type FROM races r
            JOIN tracks t ON t.id = r.track_id
            WHERE r.id = ?
        """, (target_race_id,)).fetchone()
        target_type = track_type[0] if track_type else None

        # Build race index from prior races only
        prior_race_ids = all_race_ids[:target_idx]
        race_index = {rid: idx for idx, rid in enumerate(reversed(prior_race_ids))}
        race_years = dict(conn.execute(
            f"SELECT id, year FROM races WHERE id IN ({','.join('?'*len(prior_race_ids))})",
            prior_race_ids
        ).fetchall()) if prior_race_ids else {}

        for driver_id, actual_score in actuals:
            # Fetch this driver's prior race scores
            prior_scores = conn.execute("""
                SELECT fs.race_id, fs.total_pts, t.track_type
                FROM fantasy_scores fs
                JOIN races r ON r.id = fs.race_id
                LEFT JOIN tracks t ON t.id = r.track_id
                WHERE fs.driver_id = ? AND fs.race_id IN ({})
            """.format(','.join('?'*len(prior_race_ids))),
                [driver_id] + prior_race_ids
            ).fetchall() if prior_race_ids else []

            races_for_driver = []
            for race_id, score, tt in prior_scores:
                if race_id not in race_index:
                    continue
                delta_r = race_index[race_id]
                N = max(0, target_year - race_years.get(race_id, target_year))
                races_for_driver.append({
                    "score": score,
                    "track_type": tt,
                    "delta_r": delta_r,
                    "N": N,
                })

            if not races_for_driver:
                continue

            result = model.score_driver(races_for_driver, target_type, H, phi, K)
            if result is None:
                continue

            errors.append(abs(result["p_final"] - actual_score))

    return float(np.mean(errors)) if errors else float("inf")


def main():
    print("=" * 55)
    print("  NASCAR Model Hyperparameter Tuner")
    print(f"  Grid: {len(H_VALUES)} H × {len(PHI_VALUES)} phi × {len(K_VALUES)} K"
          f" = {len(H_VALUES)*len(PHI_VALUES)*len(K_VALUES)} combos")
    print("=" * 55)

    conn = sqlite3.connect(DB_FILE)
    all_race_ids = fetch_all_races_ordered(conn)
    print(f"  Races in DB: {len(all_race_ids)}")

    best_mae  = float("inf")
    best_params = None
    total = len(H_VALUES) * len(PHI_VALUES) * len(K_VALUES)
    done  = 0

    for H, phi, K in itertools.product(H_VALUES, PHI_VALUES, K_VALUES):
        mae = evaluate_params(conn, H, phi, K, all_race_ids)
        done += 1
        print(f"  [{done}/{total}]  H={H}  phi={phi}  K={K}  → MAE={mae:.2f}")
        if mae < best_mae:
            best_mae    = mae
            best_params = {"H": H, "phi": phi, "K": K}

    conn.close()

    print(f"\n  Best: H={best_params['H']}  phi={best_params['phi']}"
          f"  K={best_params['K']}  MAE={best_mae:.2f}")

    # Merge with existing non-tuned params
    existing = load_existing_params()
    existing.update(best_params)

    # Write timestamped archive
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    archive  = f"params_{ts}.json"
    with open(archive, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Archive saved → {archive}")

    # Overwrite active params.json
    with open(PARAMS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Active params updated → {PARAMS_FILE}")

    print("\n" + "=" * 55)
    print("  Done. Run 'python query.py' to use updated params.")
    print("=" * 55)


if __name__ == "__main__":
    main()
