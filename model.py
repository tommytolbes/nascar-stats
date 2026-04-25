"""
NASCAR Driver Selection Model v2
----------------------------------
Decay-weighted blended prior + Monte Carlo optimizer + trend detection.

Usage (called by query.py):
    import model
    model.run(conn, yr, seg, tids)

Standalone tuning (run once per season):
    python tune.py
"""

import json
import sqlite3
import numpy as np
import itertools
from datetime import datetime

PARAMS_FILE = "params.json"

REQUIRED_KEYS = [
    "H", "phi", "K", "n_simulations", "random_seed",
    "n_prefilter", "min_bootstrap_samples",
    "trend_short_H", "trend_long_H",
    "trend_z_threshold", "fade_z_threshold",
]


def load_config(path=PARAMS_FILE):
    """Load and validate params.json. Raises on missing file or keys."""
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required params in {path}: {missing}")
    return cfg


def decay_weight(delta_r: int, N: int, H: float, phi: float) -> float:
    """
    Compute the decay weight for a single historical race result.

    Args:
        delta_r: Races ago (0 = most recent race on calendar).
        N:       Number of season boundaries crossed.
        H:       Half-life in races.
        phi:     Season boundary penalty (0.5 < phi < 0.9).

    Returns:
        Weight in (0, 1].
    """
    return (0.5 ** (delta_r / H)) * (phi ** N)


def fetch_driver_history(conn, yr: int, seg: int) -> dict:
    """
    Fetch all historical fantasy scores for drivers salaried in the active segment.

    Returns:
        Dict keyed by driver_id:
        {
            driver_id: {
                "name":   display_name,
                "salary": int,
                "races": [
                    {
                        "race_id":    str,
                        "score":      float,
                        "track_type": str or None,
                        "race_year":  int,
                        "delta_r":    int,   # races ago (0 = most recent)
                        "N":          int,   # season boundaries crossed
                    },
                    ...
                ],
                "scores": [float, ...],   # convenience list
            }
        }
    """
    # Ordered list of all races with results (newest first)
    all_races = conn.execute("""
        SELECT r.id, r.year
        FROM races r
        WHERE EXISTS (SELECT 1 FROM fantasy_scores fs WHERE fs.race_id = r.id)
        ORDER BY r.date DESC
    """).fetchall()

    race_index = {race_id: idx for idx, (race_id, _) in enumerate(all_races)}
    # yr is the "current" season for N calculation
    race_year  = {race_id: year for race_id, year in all_races}

    # Drivers with a salary this segment
    salaried = conn.execute("""
        SELECT ds.driver_id, d.display_name, ds.salary
        FROM driver_salaries ds
        JOIN drivers d ON d.id = ds.driver_id
        WHERE ds.year = ? AND ds.segment = ?
    """, (yr, seg)).fetchall()

    result = {}
    for driver_id, name, salary in salaried:
        rows = conn.execute("""
            SELECT fs.race_id, fs.total_pts, t.track_type
            FROM fantasy_scores fs
            JOIN races r ON r.id = fs.race_id
            LEFT JOIN tracks t ON t.id = r.track_id
            WHERE fs.driver_id = ?
        """, (driver_id,)).fetchall()

        races = []
        scores = []
        for race_id, score, track_type in rows:
            if race_id not in race_index:
                continue
            delta_r = race_index[race_id]
            N = max(0, yr - race_year[race_id])
            races.append({
                "race_id":    race_id,
                "score":      score,
                "track_type": track_type,
                "race_year":  race_year[race_id],
                "delta_r":    delta_r,
                "N":          N,
            })
            scores.append(score)

        result[driver_id] = {
            "name":   name,
            "salary": salary,
            "races":  races,
            "scores": scores,
        }

    return result
