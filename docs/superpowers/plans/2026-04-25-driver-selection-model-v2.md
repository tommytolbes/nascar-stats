# Driver Selection Model v2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple historical-average optimizer in query.py with a decay-weighted, Monte Carlo–powered optimizer plus trend detection.

**Architecture:** `model.py` handles all computation (decay weights, blended prior, Monte Carlo, trend detection) and prints output. `tune.py` performs a grid search over hyperparameters and writes results to `params.json`. `query.py` is minimally changed — `fantasy_optimizer()` is removed and replaced with `model.run()`.

**Tech Stack:** Python 3, SQLite (sqlite3), NumPy (numpy), standard library (json, itertools, datetime)

---

## Chunk 1: params.json, model.py core scoring, Monte Carlo

---

### Task 1: Create params.json with defaults

**Files:**
- Create: `params.json`
- Create: `tests/test_model.py`

- [ ] **Step 1: Create params.json**

```json
{
  "H": 10,
  "phi": 0.7,
  "K": 10,
  "n_simulations": 10000,
  "random_seed": 42,
  "n_prefilter": 20,
  "min_bootstrap_samples": 3,
  "trend_short_H": 4,
  "trend_long_H": 12,
  "trend_z_threshold": 1.0,
  "fade_z_threshold": 2.0
}
```

Save to `params.json` in the project root (same directory as `query.py`, `nascar.db`).

- [ ] **Step 2: Create tests/ directory and empty test file**

```bash
mkdir tests
touch tests/__init__.py
touch tests/test_model.py
```

- [ ] **Step 3: Commit**

```bash
git add params.json tests/__init__.py tests/test_model.py
git commit -m "feat: add params.json defaults and test skeleton"
```

---

### Task 2: model.py — load_config() with schema validation

**Files:**
- Create: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell"
python -m pytest tests/test_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Create model.py with load_config()**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model.py::test_load_config_valid tests/test_model.py::test_load_config_missing_key tests/test_model.py::test_load_config_file_not_found -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add model.py with load_config() and schema validation"
```

---

### Task 3: model.py — decay weight computation

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

The decay weight for a historical race result is:

```
W = 0.5^(Δr / H) * phi^N
```

Where Δr = races ago (0 = most recent race), N = season boundaries crossed.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model.py -k "decay_weight" -v
```

Expected: FAIL — `AttributeError: module 'model' has no attribute 'decay_weight'`

- [ ] **Step 3: Add decay_weight() to model.py**

```python
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
```

- [ ] **Step 4: Run tests to verify passage**

```bash
python -m pytest tests/test_model.py -k "decay_weight" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add decay_weight() with half-life and season boundary penalty"
```

---

### Task 4: model.py — fetch driver history from DB

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

This function pulls all historical fantasy scores for drivers with a salary in the active segment, annotates each result with `delta_r` and `N`, and returns a dict keyed by driver_id.

- [ ] **Step 1: Write failing test**

Append to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model.py -k "fetch_driver_history" -v
```

Expected: FAIL — `AttributeError: module 'model' has no attribute 'fetch_driver_history'`

- [ ] **Step 3: Add fetch_driver_history() to model.py**

```python
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
```

- [ ] **Step 4: Run tests to verify passage**

```bash
python -m pytest tests/test_model.py -k "fetch_driver_history" -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add fetch_driver_history() — annotates results with delta_r and N"
```

---

### Task 5: model.py — blended prior (P_final and σ²_final)

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model.py -k "score_driver" -v
```

Expected: FAIL — `AttributeError: module 'model' has no attribute 'score_driver'`

- [ ] **Step 3: Add score_driver() to model.py**

```python
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
                        var_general, var_final, p_final, weights, scores_all.
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
        "alpha":       alpha,
        "x_specific":  x_spec,
        "x_general":   x_gen,
        "var_specific": var_spec,
        "var_general":  var_gen,
        "var_final":   var_final,
        "p_final":     p_final,
        "scores_all":  all_scores,
        "weights_all": all_weights,
    }
```

- [ ] **Step 4: Run tests to verify passage**

```bash
python -m pytest tests/test_model.py -k "score_driver" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add score_driver() with mixture-corrected blended prior"
```

---

### Task 6: model.py — Monte Carlo simulation

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
def make_driver_scores(n_drivers=5):
    """Return a list of (name, salary, scores, weights) tuples for testing."""
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model.py -k "monte_carlo" -v
```

Expected: FAIL — `AttributeError: module 'model' has no attribute 'run_monte_carlo'`

- [ ] **Step 3: Add run_monte_carlo() to model.py**

```python
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
        w /= w.sum()
        sampling.append((d, np.array(d["scores_all"], dtype=float), w))

    # All valid combos under cap
    valid_combos = [
        combo for combo in itertools.combinations(range(len(drivers)), 4)
        if sum(drivers[i]["salary"] for i in combo) <= salary_cap
    ]

    if not valid_combos:
        return []

    # Accumulate simulated totals per combo index
    totals = {c: [] for c in valid_combos}

    for _ in range(n_simulations):
        # Sample one score per driver
        sampled = [
            float(np.random.choice(scores, p=probs))
            for _, scores, probs in sampling
        ]
        for combo in valid_combos:
            totals[combo].append(sum(sampled[i] for i in combo))

    # Compute statistics per combo
    results = []
    for combo, sims in totals.items():
        arr = np.array(sims)
        mean    = float(arr.mean())
        std     = float(arr.std())
        floor   = float(np.percentile(arr, 10))
        ceiling = float(np.percentile(arr, 90))
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
```

- [ ] **Step 4: Run tests to verify passage**

```bash
python -m pytest tests/test_model.py -k "monte_carlo" -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run all tests to confirm nothing broken**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add run_monte_carlo() with weighted bootstrap resampling"
```

---

## Chunk 2: Trend detection, run(), tune.py, query.py wiring

---

### Task 7: model.py — trend detection

**Files:**
- Modify: `model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
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
    """Driver scoring much higher recently than their baseline is hot."""
    recent = [300.0] * 4    # recent: high
    older  = [100.0] * 10   # baseline: low
    races = make_races_for_trend(recent, older)
    result = model.compute_trend(races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert result["z"] > 1.0

def test_trend_slump():
    """Driver scoring much lower recently than baseline is in a slump."""
    recent = [50.0] * 4
    older  = [250.0] * 10
    races = make_races_for_trend(recent, older)
    result = model.compute_trend(races, trend_short_H=4, trend_long_H=12, phi=0.7)
    assert result["z"] < -1.0

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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_model.py -k "trend" -v
```

Expected: FAIL — `AttributeError: module 'model' has no attribute 'compute_trend'`

- [ ] **Step 3: Add compute_trend() to model.py**

```python
def compute_trend(races: list, trend_short_H: float, trend_long_H: float,
                  phi: float) -> dict | None:
    """
    Compute trend z-score and momentum velocity for one driver.

    Uses all races (general pool only — track type ignored) with two
    different half-lives to detect hot streaks and slumps.

    Args:
        races:          Race dicts from fetch_driver_history().
        trend_short_H:  Half-life for "current form" window.
        trend_long_H:   Half-life for "baseline" window.
        phi:            Season boundary penalty (same as main model).

    Returns:
        Dict with: z, delta_z, x_short, x_long, sigma_general
        Returns None if fewer than 2 races.
    """
    if len(races) < 2:
        return None

    def weighted_avg(races_list, H):
        scores  = [r["score"] for r in races_list]
        weights = [decay_weight(r["delta_r"], r["N"], H, phi) for r in races_list]
        mean, _ = _weighted_mean_var(scores, weights)
        return mean

    x_short = weighted_avg(races, trend_short_H)
    x_long  = weighted_avg(races, trend_long_H)

    # General weighted variance (using long H as baseline)
    scores  = [r["score"] for r in races]
    weights = [decay_weight(r["delta_r"], r["N"], trend_long_H, phi) for r in races]
    _, var_general = _weighted_mean_var(scores, weights)
    sigma_general  = float(np.sqrt(var_general)) if var_general > 0 else 1.0

    z = (x_short - x_long) / sigma_general

    # delta_z: re-compute z excluding the single most recent race
    races_prev = [r for r in races if r["delta_r"] > 0]
    if len(races_prev) >= 2:
        x_short_prev = weighted_avg(races_prev, trend_short_H)
        x_long_prev  = weighted_avg(races_prev, trend_long_H)
        scores_prev  = [r["score"] for r in races_prev]
        weights_prev = [decay_weight(r["delta_r"] - 1, r["N"], trend_long_H, phi)
                        for r in races_prev]
        _, var_prev = _weighted_mean_var(scores_prev, weights_prev)
        sigma_prev  = float(np.sqrt(var_prev)) if var_prev > 0 else 1.0
        z_prev  = (x_short_prev - x_long_prev) / sigma_prev
    else:
        z_prev = z

    delta_z = z - z_prev

    return {
        "z":             round(z, 2),
        "delta_z":       round(delta_z, 2),
        "x_short":       round(x_short, 1),
        "x_long":        round(x_long, 1),
        "sigma_general": round(sigma_general, 1),
    }
```

- [ ] **Step 4: Run tests to verify passage**

```bash
python -m pytest tests/test_model.py -k "trend" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add compute_trend() with z-score and delta_z momentum velocity"
```

---

### Task 8: model.py — run() orchestration and printed output

**Files:**
- Modify: `model.py`

No new tests for `run()` — it's a printing orchestrator; correctness is validated by the functions it calls (already tested). Do a manual smoke test instead.

- [ ] **Step 1: Add run() to model.py**

```python
def _print_trend_alerts(trend_results: list, z_threshold: float,
                        fade_threshold: float) -> set:
    """
    Print trend alert section. Returns set of driver names with active alerts.
    trend_results: list of (name, z, delta_z, x_short, x_long) dicts
    """
    W = 55
    print(f"\n{'-'*W}")
    print(f"  Trend Alerts (Last ~4 races vs. baseline)")
    print(f"{'-'*W}")

    flagged_names = set()
    any_alerts = False

    for t in trend_results:
        z, dz = t["z"], t["delta_z"]
        name  = t["name"]

        if abs(z) < z_threshold:
            continue

        any_alerts = True
        pts_diff = t["x_short"] - t["x_long"]

        # Build flag emoji
        if z > fade_threshold:
            streak = "🔥⚠ "
            extra  = "  FADE RISK"
        elif z > z_threshold:
            streak = "🔥 "
            extra  = ""
        else:
            streak = "❄️  "
            extra  = ""

        if dz > 0.1:
            direction, motion = "↑", "  ACCELERATING"
        elif dz < -0.1:
            direction, motion = "↓", "  WORSENING"
        else:
            direction, motion = " ", "  STABILIZING"

        sign = "+" if pts_diff >= 0 else ""
        print(f"  {streak}{direction} {name:<22} {sign}{pts_diff:.0f} pts"
              f"  z={z:+.1f}  ΔZ={dz:+.1f}{motion}{extra}")
        flagged_names.add(name)

    if not any_alerts:
        print("  (no trend alerts this week)")

    return flagged_names


def _print_optimizer_results(results: list, flagged_names: set, top_n: int = 5):
    """Print Monte Carlo optimizer results."""
    W = 55
    print(f"\n{'-'*W}")
    print(f"  Team Optimizer — Monte Carlo (Top {top_n})")
    print(f"{'-'*W}")

    if not results:
        print("  No valid combinations found under $100.")
        return

    for rank, r in enumerate(results[:top_n], 1):
        salary    = sum(d["salary"] for d in r["combo"])
        leftover  = 100 - salary
        names_out = []
        for d in r["combo"]:
            tag = " ⚠️" if d["name"] in flagged_names else ""
            names_out.append(f"{d['name']}{tag}")
        costs = " + ".join(f"${d['salary']}" for d in r["combo"])
        print(f"\n  #{rank}  Quality: {r['quality']}  |  Mean: {r['mean']}  "
              f"|  Std: {r['std']}  |  Floor: {r['floor']}  "
              f"|  Ceil: {r['ceiling']}  |  ${salary} total  |  ${leftover} leftover")
        print(f"       {' / '.join(names_out)}")
        print(f"       {costs}")


def run(conn, yr: int, seg: int, tids: list, params_path: str = PARAMS_FILE):
    """
    Entry point called by query.py.
    Prints trend alerts then Monte Carlo optimizer results.
    """
    params = load_config(params_path)
    np.random.seed(params["random_seed"])

    H              = params["H"]
    phi            = params["phi"]
    K              = params["K"]
    n_simulations  = params["n_simulations"]
    random_seed    = params["random_seed"]
    n_prefilter    = params["n_prefilter"]
    min_samples    = params["min_bootstrap_samples"]
    trend_short_H  = params["trend_short_H"]
    trend_long_H   = params["trend_long_H"]
    z_threshold    = params["trend_z_threshold"]
    fade_threshold = params["fade_z_threshold"]

    # Determine target track type from the first segment track
    # (all tracks in a segment share a type in practice; use the first)
    track_type_row = conn.execute(
        "SELECT track_type FROM tracks WHERE id = ?", (tids[0],)
    ).fetchone()
    target_type = track_type_row[0] if track_type_row else None

    # Fetch all history
    history = fetch_driver_history(conn, yr, seg)

    # Score each driver
    scored = {}
    for driver_id, info in history.items():
        s = score_driver(info["races"], target_type, H, phi, K)
        if s is None:
            continue
        s["name"]   = info["name"]
        s["salary"] = info["salary"]
        scored[driver_id] = s

    # Trend detection
    trend_results = []
    for driver_id, info in history.items():
        t = compute_trend(info["races"], trend_short_H, trend_long_H, phi)
        if t is None:
            continue
        t["name"] = info["name"]
        trend_results.append(t)

    trend_results.sort(key=lambda x: abs(x["z"]), reverse=True)
    flagged_names = _print_trend_alerts(trend_results, z_threshold, fade_threshold)

    # Pre-filter: top n_prefilter by efficiency, min samples enforced
    eligible = [
        s for s in scored.values()
        if len(s["scores_all"]) >= min_samples and s["salary"] > 0
    ]
    eligible.sort(key=lambda x: x["p_final"] / x["salary"], reverse=True)
    pool = eligible[:n_prefilter]

    if len(pool) < 4:
        print(f"\n  (not enough eligible drivers for optimizer — need at least 4, have {len(pool)})")
        return

    mc_results = run_monte_carlo(pool, n_simulations, random_seed)
    _print_optimizer_results(mc_results, flagged_names)
```

- [ ] **Step 2: Smoke test — run query.py end-to-end**

```bash
cd "C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell"
python query.py
```

Expected: Full output including the new "Trend Alerts" section and "Team Optimizer — Monte Carlo" section at the end. The run should complete without errors.

- [ ] **Step 3: Commit**

```bash
git add model.py
git commit -m "feat: add model.run() orchestration with trend alerts and MC optimizer output"
```

---

### Task 9: Modify query.py — wire in model.run()

**Files:**
- Modify: `query.py`

- [ ] **Step 1: Add import and replace fantasy_optimizer call**

In `query.py`, make two changes:

**Change 1** — Add import near top of file (after existing imports):
```python
import model
```

**Change 2** — In `main()`, find the call to `fantasy_optimizer(conn, yr, seg, tids)` and replace it with:
```python
model.run(conn, yr, seg, tids)
```

**Change 3** — Delete the entire `fantasy_optimizer()` function (lines 262–314 in the original file).

- [ ] **Step 2: Also remove `import itertools` if no longer used**

Search `query.py` for any remaining use of `itertools`. If none, remove the import line.

```bash
grep -n "itertools" query.py
```

If only the import line appears (no other uses), remove it.

- [ ] **Step 3: Smoke test**

```bash
python query.py
```

Expected: Full report runs. The old "Team Optimizer" section is gone; the new "Trend Alerts" and "Team Optimizer — Monte Carlo" sections appear.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add query.py
git commit -m "feat: wire model.run() into query.py, remove old fantasy_optimizer()"
```

---

### Task 10: Create tune.py — grid search

**Files:**
- Create: `tune.py`

No unit tests for tune.py — it's a long-running data analysis script. Validate by running it and checking output files.

- [ ] **Step 1: Create tune.py**

```python
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
```

- [ ] **Step 2: Validate tune.py runs without errors**

```bash
python tune.py
```

Expected: Prints grid search progress table, then writes `params_YYYYMMDD_HHMM.json` and updates `params.json`. Runtime ~5-15 minutes.

If you want a quick smoke test without waiting, you can temporarily reduce the grid in the file:
```python
H_VALUES   = [10]
PHI_VALUES = [0.7]
K_VALUES   = [10]
```
Then restore after confirming it runs correctly.

- [ ] **Step 3: Commit**

```bash
git add tune.py
git commit -m "feat: add tune.py grid search for H, phi, K hyperparameters"
```

---

### Task 11: Push and finish

- [ ] **Step 1: Run full test suite one final time**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All PASSED

- [ ] **Step 2: Run query.py end-to-end**

```bash
python query.py
```

Expected: Full report with Trend Alerts and Monte Carlo optimizer sections at the end. No errors.

- [ ] **Step 3: Merge to main and push**

```bash
cd "C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR"
git merge claude/fervent-maxwell
git push origin main
```

- [ ] **Step 4: Update PICKS_GUIDE.md**

In `PICKS_GUIDE.md`, update the Troubleshooting and Step 3 sections to mention the new output sections ("Trend Alerts" and "Team Optimizer — Monte Carlo") and note that `python tune.py` can be run once per season to re-tune the model.

- [ ] **Step 5: Commit guide update and push**

```bash
git add PICKS_GUIDE.md
git commit -m "docs: update PICKS_GUIDE for model v2 output and tune.py"
git push origin main
```
