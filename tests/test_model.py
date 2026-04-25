import json
import os
import pytest
import tempfile

# Add project root to path so we can import model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model

REQUIRED_KEYS = [
    "H", "phi", "K", "n_simulations", "random_seed",
    "n_prefilter", "min_bootstrap_samples",
    "trend_short_H", "trend_long_H",
    "trend_z_threshold", "fade_z_threshold",
]

def make_params_file(data, tmp_path):
    """Write a dict to a temp params file and return the path."""
    p = tmp_path / "params.json"
    p.write_text(json.dumps(data))
    return str(p)

def valid_params():
    return {
        "H": 10, "phi": 0.7, "K": 10,
        "n_simulations": 10000, "random_seed": 42,
        "n_prefilter": 20, "min_bootstrap_samples": 3,
        "trend_short_H": 4, "trend_long_H": 12,
        "trend_z_threshold": 1.0, "fade_z_threshold": 2.0,
    }

def test_load_config_valid(tmp_path):
    path = make_params_file(valid_params(), tmp_path)
    cfg = model.load_config(path)
    assert cfg["H"] == 10
    assert cfg["phi"] == 0.7

def test_load_config_missing_key(tmp_path):
    data = valid_params()
    del data["H"]
    path = make_params_file(data, tmp_path)
    with pytest.raises(ValueError, match="Missing required params"):
        model.load_config(path)

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        model.load_config("nonexistent_params.json")


def test_decay_weight_most_recent():
    # Race from 0 races ago, same season: weight should be 1.0
    w = model.decay_weight(delta_r=0, N=0, H=10, phi=0.7)
    assert w == pytest.approx(1.0)

def test_decay_weight_half_life():
    # Race from exactly H races ago, same season: weight should be 0.5
    w = model.decay_weight(delta_r=10, N=0, H=10, phi=0.7)
    assert w == pytest.approx(0.5)

def test_decay_weight_season_boundary():
    # One season boundary crossed applies phi multiplier
    w_no_boundary = model.decay_weight(delta_r=10, N=0, H=10, phi=0.7)
    w_boundary    = model.decay_weight(delta_r=10, N=1, H=10, phi=0.7)
    assert w_boundary == pytest.approx(w_no_boundary * 0.7)

def test_decay_weight_two_seasons():
    # Two boundaries: phi^2
    w = model.decay_weight(delta_r=0, N=2, H=10, phi=0.7)
    assert w == pytest.approx(0.7 ** 2)


import sqlite3

def make_test_db(tmp_path):
    """Build a minimal in-memory-style DB at tmp_path/test.db."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE drivers (id INTEGER PRIMARY KEY, display_name TEXT, first_name TEXT, last_name TEXT);
        CREATE TABLE tracks (id INTEGER PRIMARY KEY, track_type TEXT, full_name TEXT);
        CREATE TABLE races (id TEXT PRIMARY KEY, year INTEGER, date TEXT, track_id INTEGER, race_num INTEGER);
        CREATE TABLE fantasy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT, driver_id INTEGER, total_pts REAL
        );
        CREATE TABLE driver_salaries (
            driver_id INTEGER, year INTEGER, segment INTEGER, salary INTEGER
        );

        INSERT INTO drivers VALUES (1, 'Driver A', 'Driver', 'A');
        INSERT INTO drivers VALUES (2, 'Driver B', 'Driver', 'B');
        INSERT INTO tracks VALUES (10, 'intermediate', 'Test Track');
        INSERT INTO races VALUES ('r1', 2025, '2025-01-01', 10, 1);
        INSERT INTO races VALUES ('r2', 2025, '2025-02-01', 10, 2);
        INSERT INTO races VALUES ('r3', 2026, '2026-01-01', 10, 3);
        INSERT INTO fantasy_scores (race_id, driver_id, total_pts) VALUES ('r1', 1, 100.0);
        INSERT INTO fantasy_scores (race_id, driver_id, total_pts) VALUES ('r2', 1, 150.0);
        INSERT INTO fantasy_scores (race_id, driver_id, total_pts) VALUES ('r3', 1, 200.0);
        INSERT INTO driver_salaries VALUES (1, 2026, 1, 25);
        INSERT INTO driver_salaries VALUES (2, 2026, 1, 10);
    """)
    conn.commit()
    return conn

def test_fetch_driver_history_returns_scores(tmp_path):
    conn = make_test_db(tmp_path)
    history = model.fetch_driver_history(conn, yr=2026, seg=1)
    # Driver 1 has 3 scores; Driver 2 has none
    assert 1 in history
    assert len(history[1]["scores"]) == 3

def test_fetch_driver_history_delta_r(tmp_path):
    conn = make_test_db(tmp_path)
    history = model.fetch_driver_history(conn, yr=2026, seg=1)
    # Most recent race (r3) should have delta_r=0
    most_recent = min(history[1]["races"], key=lambda x: x["delta_r"])
    assert most_recent["delta_r"] == 0

def test_fetch_driver_history_season_N(tmp_path):
    conn = make_test_db(tmp_path)
    history = model.fetch_driver_history(conn, yr=2026, seg=1)
    races = {r["race_id"]: r for r in history[1]["races"]}
    # r3 is 2026: N=0; r1 and r2 are 2025: N=1
    assert races["r3"]["N"] == 0
    assert races["r1"]["N"] == 1
    assert races["r2"]["N"] == 1
