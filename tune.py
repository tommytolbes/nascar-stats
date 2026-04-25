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

# Grid search space (140 total combinations)
H_VALUES   = [4, 6, 8, 10, 12, 16, 20]   # half-life in races; 4=very recent, 20=long memory
PHI_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]   # season boundary penalty; <0.5 too severe, >0.9 too lenient
K_VALUES   = [5, 8, 10, 15]               # specialist saturation; at n=K, alpha=1

WARMUP = 10  # skip first N races — not enough prior history to make useful predictions


def load_existing_params(path=PARAMS_FILE):
    """Load existing params.json to preserve non-tuned keys (n_simulations, etc.)."""
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


def preload_data(conn):
    """
    Load all needed data from DB into memory for fast CV.

    Returns:
        race_meta:     {race_id: {"year": int, "track_type": str}}
        race_results:  {race_id: [(driver_id, total_pts), ...]}
        driver_scores: {driver_id: [(race_id, total_pts), ...]}
    """
    race_meta = {}
    for race_id, year, track_type in conn.execute("""
        SELECT r.id, r.year, t.track_type
        FROM races r
        LEFT JOIN tracks t ON t.id = r.track_id
    """).fetchall():
        race_meta[race_id] = {"year": year, "track_type": track_type}

    race_results  = {}
    driver_scores = {}
    for race_id, driver_id, pts in conn.execute(
        "SELECT race_id, driver_id, total_pts FROM fantasy_scores"
    ).fetchall():
        race_results.setdefault(race_id, []).append((driver_id, pts))
        driver_scores.setdefault(driver_id, []).append((race_id, pts))

    return race_meta, race_results, driver_scores


def evaluate_params(H, phi, K, all_race_ids, race_meta, race_results, driver_scores):
    """
    Leave-one-out time-series CV: for each race (after warm-up),
    predict each driver's score using all prior races, compute MAE.

    All data is pre-loaded in memory — no DB queries during the grid search.

    Returns mean absolute error across all predictions.
    """
    errors = []

    for target_idx in range(WARMUP, len(all_race_ids)):
        target_race_id = all_race_ids[target_idx]

        if target_race_id not in race_meta or target_race_id not in race_results:
            continue

        meta        = race_meta[target_race_id]
        target_year = meta["year"]
        target_type = meta["track_type"]
        actuals     = race_results[target_race_id]

        prior_race_ids = all_race_ids[:target_idx]
        prior_set      = set(prior_race_ids)
        race_index     = {rid: idx for idx, rid in enumerate(reversed(prior_race_ids))}

        for driver_id, actual_score in actuals:
            races_for_driver = []
            for race_id, score in driver_scores.get(driver_id, []):
                if race_id not in prior_set:
                    continue
                delta_r    = race_index[race_id]
                race_year  = race_meta[race_id]["year"]
                N          = max(0, target_year - race_year)
                track_type = race_meta[race_id]["track_type"]
                races_for_driver.append({
                    "score":      score,
                    "track_type": track_type,
                    "delta_r":    delta_r,
                    "N":          N,
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

    print("  Pre-loading race and score data...")
    race_meta, race_results, driver_scores = preload_data(conn)
    conn.close()
    print(f"  Loaded {len(race_meta)} races, {sum(len(v) for v in race_results.values())} scores.")

    best_mae    = float("inf")
    best_params = None
    total = len(H_VALUES) * len(PHI_VALUES) * len(K_VALUES)
    done  = 0

    for H, phi, K in itertools.product(H_VALUES, PHI_VALUES, K_VALUES):
        mae = evaluate_params(H, phi, K, all_race_ids, race_meta, race_results, driver_scores)
        done += 1
        print(f"  [{done}/{total}]  H={H}  phi={phi}  K={K}  → MAE={mae:.2f}")
        if mae < best_mae:
            best_mae    = mae
            best_params = {"H": H, "phi": phi, "K": K}

    print(f"\n  Best: H={best_params['H']}  phi={best_params['phi']}"
          f"  K={best_params['K']}  MAE={best_mae:.2f}")

    # Merge with existing non-tuned params (preserves n_simulations, random_seed, etc.)
    existing = load_existing_params()
    existing.update(best_params)

    # Write timestamped archive
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    archive = f"params_{ts}.json"
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
