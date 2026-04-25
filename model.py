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
