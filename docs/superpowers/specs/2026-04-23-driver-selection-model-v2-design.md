# Driver Selection Model v2 — Design Spec
**Date:** 2026-04-23  
**Project:** Braswell's Fantasy NASCAR  
**Status:** Approved for implementation

---

## Overview

The current optimizer in `query.py` ranks 4-driver combos by a simple historical average of fantasy points, with no awareness of recency, 2026 current-season performance, or driver variance. This spec describes a replacement model that incorporates exponential decay weighting, a blended track-specific/general prior, Monte Carlo simulation with variance-aware combo ranking, and trend detection with momentum velocity.

---

## Architecture

Four files with separated responsibilities:

| File | Role |
|---|---|
| `tune.py` | Grid search over H, φ, K → writes timestamped archive + active `params.json` |
| `params.json` | Single source of truth for all model hyperparameters |
| `model.py` | Reads `params.json`, computes blended prior scores, runs Monte Carlo, returns ranked combos and trend alerts |
| `query.py` | Unchanged except `fantasy_optimizer()` is replaced with a call to `model.py` |

### params.json structure

```json
{
  "H": 10,
  "phi": 0.7,
  "K": 10,
  "n_simulations": 10000,
  "random_seed": 42,
  "trend_short_H": 4,
  "trend_long_H": 12,
  "trend_z_threshold": 1.0,
  "fade_z_threshold": 2.0
}
```

### Schema validation

`model.py` validates `params.json` on every load via a `load_config()` function that checks for all required keys and raises a `ValueError` with a clear message if any are missing. This prevents silent failures from a partial write by `tune.py`.

### Model versioning

On every run, `tune.py` writes two files:
- `params_YYYYMMDD_HHMM.json` — timestamped archive, never overwritten
- `params.json` — active parameters, overwritten each run

This creates an audit trail and allows rollback to any prior tuning session.

### Reproducibility

`model.py` calls `np.random.seed(params["random_seed"])` at the start of the Monte Carlo function. The seed is stored in `params.json` so it can be changed deliberately without touching code.

---

## Per-Driver Scoring

For each driver with a salary in the active segment, `model.py` computes the following:

### Step 1 — Decay weight per historical race

$$W = 0.5^{(\Delta r / H)} \cdot \phi^{(N)}$$

- `Δr` = number of Cup races ago this result occurred, counting all races on the calendar (not just this driver's starts)
- `N` = number of season boundaries crossed (e.g., a 2025 result when projecting for 2026 → N=1)
- `H` = half-life in races (from `params.json`)
- `φ` = season boundary penalty constant, 0.5 < φ < 0.9 (from `params.json`)

### Step 2 — Two decay-weighted averages

$$\bar{x}_{specific} = \frac{\sum W_i \cdot s_i}{\sum W_i} \quad \text{(races at the same track type as the segment)}$$

$$\bar{x}_{general} = \frac{\sum W_i \cdot s_i}{\sum W_i} \quad \text{(all races)}$$

### Step 3 — Confidence weight

$$\alpha = \min\!\left(1,\ \frac{n}{K}\right)$$

- `n` = raw (unweighted) count of the driver's starts at the segment's track type
- `K` = saturation constant (from `params.json`); α approaches 1 as n grows, transitioning from general form to track specialist automatically

### Step 4 — Blended projection

$$P_{final} = \alpha \cdot \bar{x}_{specific} + (1-\alpha) \cdot \bar{x}_{general}$$

### Step 5 — Blended variance (mixture-corrected)

$$\sigma^2_{final} = \alpha\sigma^2_{spec} + (1-\alpha)\sigma^2_{gen} + \alpha(1-\alpha)(\bar{x}_{spec} - \bar{x}_{gen})^2$$

The third term accounts for the between-component variance of the mixture distribution. Omitting it underestimates total variance, making the Monte Carlo simulation overconfident. It is largest when the two pool means differ significantly (e.g., a true track specialist).

Weighted variance for each pool:
$$\sigma^2 = \frac{\sum W_i \cdot (s_i - \bar{x})^2}{\sum W_i}$$

---

## Monte Carlo Simulation

### Pre-filter

Before simulating, filter to the top 20 drivers by `P_final / Salary` (efficiency), not raw `P_final`. Filtering by efficiency preserves low-salary, high-ceiling "punt" drivers who are essential to fitting elite drivers under the salary cap.

### Sampling method

Weighted bootstrap resampling from actual historical scores. For each driver:

```
scores = [s₁, s₂, ...]      # actual historical fantasy scores
weights = [W₁, W₂, ...]     # decay weights
probs = weights / sum(weights)
sampled = np.random.choice(scores, p=probs)
```

Bootstrap resampling is used instead of parametric (normal) sampling to preserve the real shape of each driver's distribution, including heavy left tails from DNFs and crashes.

### Simulation loop

For each of `n_simulations` iterations:
1. Sample one score per driver from their weighted empirical distribution
2. Evaluate all valid 4-driver combos under the $100 salary cap
3. Record the simulated total for each combo

### Per-combo statistics

After all simulations, compute for each combo:
- **Mean** — expected total; primary sort key
- **Std Dev** — risk/variance indicator
- **Floor** — 10th percentile simulated total
- **Ceiling** — 90th percentile simulated total
- **Quality** = Mean / Std Dev (Sharpe-style risk-adjusted metric)

High Quality + high Mean = consistent cash-game lineup.  
High Mean + low Quality = tournament/GPP lineup (boom-or-bust).

### Output format

```
#1  Quality: 6.1  |  Mean: 412  |  Std: 68  |  Floor: 301  |  Ceil: 521  |  $95
    Hamlin / Byron / Chastain ⚠️ / Dillon  —  $27+$33+$30+$5
```

Drivers flagged with a trend alert (slump or fade risk) are marked ⚠️ inline.

---

## Trend Detection

Trend detection runs in the general pool (all races, not track-specific) using two separate invocations of the decay model.

### Base trend score

$$z = \frac{\bar{x}_{short} - \bar{x}_{long}}{\sigma_{general}}$$

- `x̄_short` = decay-weighted average using `trend_short_H` (default: 4 races)
- `x̄_long` = decay-weighted average using `trend_long_H` (default: 12 races)
- `σ_general` = driver's overall weighted standard deviation

Normalizing by σ accounts for driver-specific variance: a boom-or-bust driver requires a larger swing to be flagged than a consistently average one.

### Momentum velocity

$$\Delta Z = Z_{current} - Z_{previous}$$

`Z_previous` is computed by re-running the trend model with the single most recent race excluded. No state file is required; the computation is fully self-contained. A large positive ΔZ indicates an accelerating trend — the highest-value opportunity in DFS before salary adjustments catch up.

### Flags

| Condition | Label |
|---|---|
| z > `trend_z_threshold` | 🔥 Hot streak |
| z < −`trend_z_threshold` | ❄️ Slump |
| ΔZ > 0 | ↑ Accelerating |
| ΔZ < 0 | ↓ Worsening |
| ΔZ ≈ 0 | Stabilizing |
| z > `fade_z_threshold` | ⚠ FADE RISK |

### Output format

Shown as a dedicated section in `query.py` output, before the optimizer:

```
Trend Alerts (Last ~4 races vs. baseline)
-------------------------------------------------------
🔥↑ William Byron     +34 pts  z=+1.8  ΔZ=+0.4  ACCELERATING
🔥⚠  Tyler Reddick    +52 pts  z=+2.3  ΔZ=+0.1  FADE RISK
❄️↓ Ross Chastain     -41 pts  z=-2.1  ΔZ=-0.3  WORSENING
❄️   Kyle Busch       -39 pts  z=-1.9  ΔZ=+0.1  STABILIZING
```

---

## Hyperparameter Tuning (tune.py)

### Cross-validation method

Leave-one-out time-series cross-validation. For each race in the historical dataset (in chronological order):
1. Use all races prior to this race to compute `P_final` per driver
2. Compare `P_final` to actual fantasy score for this race
3. Accumulate absolute error

The combination of H, φ, K that minimizes mean absolute error (MAE) across all predictions is selected.

### Grid search space

| Parameter | Values searched |
|---|---|
| H | 4, 6, 8, 10, 12, 16, 20 |
| φ | 0.5, 0.6, 0.7, 0.8, 0.9 |
| K | 5, 8, 10, 15 |

Total combinations: 7 × 5 × 4 = 140 grid points.

### Output

Winning parameters written to:
- `params_YYYYMMDD_HHMM.json` — timestamped archive
- `params.json` — overwrites active parameters

Non-tuned parameters (`n_simulations`, `random_seed`, trend thresholds) are preserved from the existing `params.json` and not overwritten by `tune.py`.

---

## query.py Changes

Minimal. The existing `fantasy_optimizer()` function is removed and replaced with:

```python
import model
model.run(conn, yr, seg, tids)
```

`model.run()` prints the trend alerts section and the optimizer combos section directly, matching the existing output style of `query.py`.

All other sections of `query.py` (database summary, recent form, track history, track type specialists) remain untouched.

---

## Out of Scope

- Storing projections in the database (deferred — no current need)
- UI or web-based output
- Integration with external DFS salary sources
- Automated weekly re-tuning (tune.py is run manually)
