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


def _weighted_mean_var(scores: list, weights: list) -> tuple:
    """Return (weighted_mean, weighted_variance) for parallel lists."""
    w = np.array(weights, dtype=float)
    s = np.array(scores, dtype=float)
    w_sum = w.sum()
    if w_sum == 0:
        return 0.0, 0.0
    mean = (w * s).sum() / w_sum
    var  = (w * (s - mean) ** 2).sum() / w_sum
    return float(mean), float(var)


def score_driver(races: list, target_type: str,
                 H: float, phi: float, K: int) -> dict | None:
    """
    Compute blended prior projection for one driver.

    Args:
        races:       List of race dicts from fetch_driver_history().
        target_type: Track type for the upcoming segment (e.g. "intermediate").
        H, phi, K:   Hyperparameters from params.json.

    Returns:
        Dict with keys: alpha, x_specific, x_general, var_specific,
                        var_general, var_final, p_final, weights_all, scores_all.
        Returns None if driver has no historical scores at all.
    """
    if not races:
        return None

    all_scores  = []
    all_weights = []
    spec_scores  = []
    spec_weights = []

    for r in races:
        w = decay_weight(r["delta_r"], r["N"], H, phi)
        all_scores.append(r["score"])
        all_weights.append(w)
        if r["track_type"] == target_type:
            spec_scores.append(r["score"])
            spec_weights.append(w)

    x_gen, var_gen   = _weighted_mean_var(all_scores, all_weights)
    x_spec, var_spec = _weighted_mean_var(spec_scores, spec_weights) if spec_scores else (x_gen, var_gen)

    n     = len(spec_scores)
    alpha = min(1.0, n / K)

    p_final  = alpha * x_spec + (1 - alpha) * x_gen
    var_final = (
        alpha * var_spec
        + (1 - alpha) * var_gen
        + alpha * (1 - alpha) * (x_spec - x_gen) ** 2
    )

    return {
        "alpha":        alpha,
        "x_specific":   x_spec,
        "x_general":    x_gen,
        "var_specific": var_spec,
        "var_general":  var_gen,
        "var_final":    var_final,
        "p_final":      p_final,
        "scores_all":   all_scores,
        "weights_all":  all_weights,
    }


def run_monte_carlo(drivers: list, n_simulations: int,
                    random_seed: int, salary_cap: int = 100) -> list:
    """
    Run Monte Carlo simulation to rank 4-driver combos.

    Args:
        drivers:       List of dicts, each with keys:
                         name, salary, p_final, scores_all, weights_all
        n_simulations: Number of simulation iterations.
        random_seed:   RNG seed for reproducibility.
        salary_cap:    Maximum combined salary (default 100).

    Returns:
        List of combo result dicts sorted by mean descending:
        [
            {
                "mean":    float,
                "std":     float,
                "floor":   float,   # 10th percentile
                "ceiling": float,   # 90th percentile
                "quality": float,   # mean / std (Sharpe ratio)
                "combo":   [driver_dict, ...],  # 4 driver dicts
            },
            ...
        ]
    """
    np.random.seed(random_seed)

    # Pre-build sampling distributions per driver
    sampling = []
    for d in drivers:
        w = np.array(d["weights_all"], dtype=float)
        w_sum = w.sum()
        if w_sum == 0:
            raise ValueError(f"Driver '{d['name']}' has all-zero weights — cannot sample")
        w /= w_sum
        sampling.append((d, np.array(d["scores_all"], dtype=float), w))

    # All valid combos under cap
    valid_combos = [
        combo for combo in itertools.combinations(range(len(drivers)), 4)
        if sum(drivers[i]["salary"] for i in combo) <= salary_cap
    ]

    if not valid_combos:
        return []

    # Generate all samples at once: shape (n_simulations, n_drivers)
    all_sampled = np.column_stack([
        np.random.choice(scores, size=n_simulations, p=probs)
        for _, scores, probs in sampling
    ])

    # Compute stats per combo using array slicing
    results = []
    for combo in valid_combos:
        combo_totals = all_sampled[:, list(combo)].sum(axis=1)
        mean    = float(combo_totals.mean())
        std     = float(combo_totals.std())
        floor   = float(np.percentile(combo_totals, 10))
        ceiling = float(np.percentile(combo_totals, 90))
        quality = round(mean / std, 2) if std > 0 else 0.0
        results.append({
            "mean":    round(mean, 1),
            "std":     round(std, 1),
            "floor":   round(floor, 1),
            "ceiling": round(ceiling, 1),
            "quality": quality,
            "combo":   [drivers[i] for i in combo],
        })

    results.sort(key=lambda x: x["mean"], reverse=True)
    return results
