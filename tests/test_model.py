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
    # Driver 2 is salaried but has no results — should appear with empty lists
    assert 2 in history
    assert history[2]["scores"] == []

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


def test_score_driver_general_only():
    """With zero track-specific starts, result equals general decay-weighted avg."""
    races = [
        {"score": 100.0, "track_type": "intermediate", "delta_r": 0, "N": 0},
        {"score": 200.0, "track_type": "intermediate", "delta_r": 5, "N": 0},
    ]
    # target_type="road_course" so n_specific=0 → alpha=0 → P_final = x_general
    result = model.score_driver(
        races, target_type="road_course",
        H=10, phi=0.7, K=10
    )
    assert result["alpha"] == pytest.approx(0.0)
    assert result["p_final"] == pytest.approx(result["x_general"])

def test_score_driver_full_specialist():
    """With n >= K starts, alpha == 1 and P_final equals x_specific."""
    races = [
        {"score": 150.0, "track_type": "superspeedway", "delta_r": i, "N": 0}
        for i in range(10)
    ]
    result = model.score_driver(
        races, target_type="superspeedway",
        H=10, phi=0.7, K=10
    )
    assert result["alpha"] == pytest.approx(1.0)
    assert result["p_final"] == pytest.approx(result["x_specific"])

def test_score_driver_variance_includes_mixture_term():
    """Variance should be >= weighted avg of component variances when means differ."""
    races = (
        [{"score": 300.0, "track_type": "superspeedway", "delta_r": i, "N": 0} for i in range(5)] +
        [{"score": 50.0,  "track_type": "intermediate",  "delta_r": i + 5, "N": 0} for i in range(5)]
    )
    result = model.score_driver(
        races, target_type="superspeedway",
        H=10, phi=0.7, K=10
    )
    # When means differ, mixture variance > plain weighted variance
    plain = result["alpha"] * result["var_specific"] + (1 - result["alpha"]) * result["var_general"]
    assert result["var_final"] >= plain - 1e-9  # >= with float tolerance

def test_score_driver_insufficient_data():
    """Driver with no scores returns None."""
    result = model.score_driver([], target_type="intermediate", H=10, phi=0.7, K=10)
    assert result is None


def make_driver_scores(n_drivers=5):
    """Return a list of driver dicts for Monte Carlo testing."""
    drivers = []
    for i in range(n_drivers):
        scores  = [100.0 + i * 10] * 5
        weights = [1.0] * 5
        drivers.append({
            "name":        f"Driver {i}",
            "salary":      20 + i,
            "p_final":     100.0 + i * 10,
            "scores_all":  scores,
            "weights_all": weights,
        })
    return drivers

def test_monte_carlo_returns_combos():
    drivers = make_driver_scores(5)
    results = model.run_monte_carlo(drivers, n_simulations=100, random_seed=42)
    assert len(results) > 0
    assert "mean" in results[0]
    assert "std" in results[0]
    assert "floor" in results[0]
    assert "ceiling" in results[0]
    assert "quality" in results[0]
    assert "combo" in results[0]

def test_monte_carlo_salary_cap():
    """No combo in results should exceed $100 total salary."""
    drivers = make_driver_scores(8)
    results = model.run_monte_carlo(drivers, n_simulations=100, random_seed=42)
    for r in results:
        total = sum(d["salary"] for d in r["combo"])
        assert total <= 100

def test_monte_carlo_reproducible():
    """Same seed → same top combo mean."""
    drivers = make_driver_scores(6)
    r1 = model.run_monte_carlo(drivers, n_simulations=200, random_seed=42)
    r2 = model.run_monte_carlo(drivers, n_simulations=200, random_seed=42)
    assert r1[0]["mean"] == pytest.approx(r2[0]["mean"])

def test_monte_carlo_zero_weight_raises():
    """Driver with all-zero weights should raise ValueError."""
    drivers = make_driver_scores(5)
    drivers[2]["weights_all"] = [0.0] * len(drivers[2]["weights_all"])
    with pytest.raises(ValueError, match="Driver"):
        model.run_monte_carlo(drivers, n_simulations=100, random_seed=42)


def make_races_for_trend(recent_scores, older_scores):
    """Build a race list where recent scores have low delta_r, older have high delta_r."""
    races = []
    for i, score in enumerate(recent_scores):
        races.append({"score": score, "track_type": "intermediate",
                      "delta_r": i, "N": 0})
    for j, score in enumerate(older_scores):
        races.append({"score": score, "track_type": "intermediate",
                      "delta_r": len(recent_scores) + j, "N": 0})
    return races

def test_trend_hot_streak():
    """Driver scoring higher recently than baseline has positive z."""
    recent = [300.0] * 4    # recent: high
    older  = [100.0] * 10   # baseline: low
    races = make_races_for_trend(recent, older)
    result = model.compute_trend(races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert result["z"] > 0  # hot driver has positive z

def test_trend_slump():
    """Driver scoring lower recently than baseline has negative z."""
    recent = [50.0] * 4
    older  = [250.0] * 10
    races = make_races_for_trend(recent, older)
    result = model.compute_trend(races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert result["z"] < 0  # slumping driver has negative z

def test_trend_z_crosses_threshold_with_strong_signal():
    """With a very short H, a clear step-change drives z above the flag threshold."""
    recent = [300.0] * 4
    older  = [100.0] * 10
    races = make_races_for_trend(recent, older)
    # trend_short_H=1 focuses almost entirely on the most recent race
    result = model.compute_trend(races, trend_short_H=1, trend_long_H=30, phi=0.7)
    assert result["z"] > 1.0

def test_trend_hot_beats_slump():
    """Hot driver always has higher z than slumped driver."""
    hot_races   = make_races_for_trend([300.0] * 4, [100.0] * 10)
    slump_races = make_races_for_trend([100.0] * 4, [300.0] * 10)
    hot   = model.compute_trend(hot_races,   trend_short_H=4, trend_long_H=12, phi=0.7)
    slump = model.compute_trend(slump_races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert hot["z"] > slump["z"]

def test_trend_delta_z():
    """delta_z is computed and is a float."""
    races = make_races_for_trend([200.0] * 4, [150.0] * 10)
    result = model.compute_trend(races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert isinstance(result["delta_z"], float)

def test_trend_insufficient_data():
    """Returns None when fewer than 2 races exist."""
    result = model.compute_trend(
        [{"score": 100.0, "track_type": "x", "delta_r": 0, "N": 0}],
        trend_short_H=4, trend_long_H=12, phi=0.7
    )
    assert result is None
