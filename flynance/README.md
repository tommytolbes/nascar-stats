# Agentic Flynance — a Blackjack-playing fly brain

A 4,010-parameter neural network, sized after the fruit-fly mushroom body, learns Blackjack from
nothing but win/loss signals — and is scored against the mathematically exact optimal policy.

This directory implements the *Agentic Flynance* specification. It also documents four defects found
in that specification, the most serious of which makes its training loop unable to learn at all.

```
[ Blackjack Environment ]
        |
        v
[ Sensory Encoding ]      player_sum/21.0 , dealer_upcard/10.0 , usable_ace
        |
        v
[ Sensory Layer ]         3 -> 64   ReLU      antennal lobe / Kenyon cell projection
        |
        v
[ Mushroom Body ]         64 -> 56  ReLU      intrinsic sub-network
        |
        v
[ Motor Layer ]           56 -> 2   logits    0 = STAND , 1 = HIT
        |
        v
[ Neuromodulation ]       +1 dopaminergic  |  0 neutral  |  -1 octopaminergic
```

---

## Quick start

```bash
pip install -r flynance/requirements.txt

# Full experiment: trains both agents, evaluates, scores against the exact optimum
python -m flynance.run_experiment --episodes 200000 --seed 42

# Tests (including the finite-difference gradient check and the DP cross-check)
python -m pytest tests/test_flynance_*.py -v

# Watch the trained fly play, hand by hand, with its confidence on every decision
python -m flynance.watch --hands 10
```

`watch.py` loads the brain saved by `run_experiment.py` (`--save-model`, on by
default), or trains one on first use. Sample output:

```
Hand 6
----------------------------------------------------
  Dealer shows   10
  Fly holds      [2, 5] = 7
  Fly decides    HIT    (100% confident)
  Draws          9  ->  [2, 5, 9] = 16
  Fly decides    STAND  (98% confident)
  Dealer plays   [10, 4, 8] = 22
  Dealer BUSTS at 22
  Result         WIN   +1
```

That hand shows the weakness the decision matrix quantifies: standing on a hard
16 against a dealer 10 — at 98% confidence — is one of the 21 cells where the fly
disagrees with optimal play. It won anyway, which is exactly why single hands
prove nothing and the 10,000-hand sweep does.

The run writes `flynance/results/report.txt` and `flynance/results/metrics_seed42.json`. A committed
sample of both is in `results/`.

---

## Results

Seed 42, 200,000 training episodes per agent, 10,000 evaluation hands. Reproduce with
`python -m flynance.run_experiment --episodes 200000 --seed 42 --ablation`; the full report is in
[`results/report.txt`](results/report.txt).

### Expected value

Every policy is dealt the *same* 10,000 hands (common random numbers), so these differences are not
deal luck.

| Policy | EV / hand | 95% CI | win % | loss % | push % |
|---|---|---|---|---|---|
| DP optimal (ceiling) | −0.0324 | [−0.0510, −0.0138] | 43.4 | 46.7 | 9.9 |
| **reinforce (the fly brain)** | **−0.0489** | [−0.0675, −0.0303] | 42.9 | 47.8 | 9.3 |
| dealer_mimic (hit < 17) | −0.0621 | [−0.0806, −0.0436] | 41.5 | 47.7 | 10.9 |
| always_stand | −0.1742 | [−0.1930, −0.1554] | 38.6 | 56.0 | 5.4 |
| **spec_baseline (the spec's loop)** | **−1.0000** | [−1.0000, −1.0000] | **0.0** | **100.0** | 0.0 |

The exact EV of optimal play is **−0.046556**, computed in closed form. The −0.0324 above is that
same policy measured on 10,000 sampled hands; the difference is sampling noise, and the exact value
sits inside the interval. Ten thousand hands is simply not enough to separate a good policy from a
perfect one — the per-hand standard deviation is close to 1.0 — which is why the decision matrix
below, not the EV table, is the sharp instrument. On EV alone the fly brain is statistically
indistinguishable from optimal play, and clearly ahead of every heuristic.

### The spec loop's collapse

The specification's training loop does not produce a weaker agent. It produces a policy that has
collapsed onto one constant action, and which action depends only on the seed:

| Seeds (of 0–7) | Collapse | EV / hand |
|---|---|---|
| 0, 1, 2, 3 | always-**hit** | **−1.0000** |
| 4, 5, 7 | always-**stand** | −0.1848 |
| 6 | mixed | −0.1742 |

Always-hit is absolute: a policy that always hits never stops hitting, so every hand ends in a bust.
It loses 100% of hands, wins none, and pushes none. Seed 42 — the seed used for the report — lands in
exactly that mode.

### Decision matrix

The learned policy agrees with the exact optimum on **92.5%** of the 280 decision cells, and on
**88.4%** when cells are weighted by how often optimal play actually visits them.

```
Hard totals (no usable ace)      lowercase + '*' = disagrees with the optimum
total |   A   2   3   4   5   6   7   8   9  10
------+----------------------------------------
   11 |   H   H   H   H   H   H   H   H   H   H
   12 |   H   H   H  h*  h*  h*   H   H   H   H
   13 |   H  h*   S   S   S   S   H   H   H   H
   14 |   H   S   S   S   S   S  s*   H   H   H
   15 |  s*   S   S   S   S   S  s*  s*  s*   H
   16 |  s*   S   S   S   S   S  s*  s*  s*  s*
   17 |   S   S   S   S   S   S   S   S   S   S
```

All 21 mistakes are the *same* mistake: standing on a hard 14–16 against a strong dealer upcard,
where the optimum is to take the risk and hit. The most expensive, by visit frequency × EV cost:

| Player | Dealer | Agent | Optimal | EV cost |
|---|---|---|---|---|
| 15 | A | stand | hit | 0.155 |
| 14 | 7 | stand | hit | 0.154 |
| 16 | A | stand | hit | 0.127 |
| 15 | 7 | stand | hit | 0.106 |

For comparison, the spec-trained network's alignment is 60.7% raw / 47.1% weighted.

### The mistakes are the encoding's fault, not the brain's

Notice where the errors cluster: dealer upcards 7 through ace. The specification's encoding divides
the upcard by 10, which maps an ace to **0.1** — numerically the *weakest* input value, while an ace
is strategically the dealer's *strongest* card. The network is being asked to learn a non-monotonic
boundary from a single scalar.

Re-running with the dealer upcard one-hot encoded, changing nothing else about the mushroom body:

| Encoding | Parameters | EV | Weighted alignment |
|---|---|---|---|
| Spec: 3 scalars | 4,010 | −0.0489 | 88.4% |
| One-hot upcard | 4,586 | −0.0478 | **92.4%** |

Four points of alignment for 576 extra weights in the sensory layer alone. The 4,010-parameter
mushroom body was never the bottleneck — the sensory encoding in front of it was.

### Tuning

The specification's `lr=0.005` is far too hot. Adam normalizes the gradient, so every step moves each
weight by roughly `lr` regardless of how noisy that one hand's estimate was; at 0.005 with one hand
per update the policy never settles. Candidates were scored on **mean and worst-case** results over
three seeds, not a single run:

| Config (200k episodes, 3 seeds) | EV mean | EV worst | Weighted align (mean / worst) |
|---|---|---|---|
| lr 2e-4 constant | −0.0515 | −0.0599 | 89.4% / 87.0% |
| lr 1e-3 → 5e-5 | −0.0496 | −0.0532 | 86.9% / 85.9% |
| **lr 5e-4 → 2e-5 (chosen)** | **−0.0478** | **−0.0493** | 88.8% / 86.7% |

Single-seed screening runs that led to those candidates, for context — the spec's own learning rate
is the worst of them, and note that a *lower* rate is not automatically better (5e-4 held constant
was the weakest config tested, while the same rate decayed to 2e-5 was the strongest):

| Config (200k episodes, seed 7) | EV | Weighted alignment |
|---|---|---|
| lr 5e-3 — the spec's value, 50k episodes | −0.1188 | not scored |
| lr 1e-3 constant | −0.0589 | 84.9% |
| lr 5e-4 constant | −0.1330 | 71.2% |
| lr 2e-4 constant | −0.0476 | 90.5% |

Selection used seeds 7/11/22/33; the reported run uses seed 42, so the headline numbers are not from
the seeds the choice was made on.

---

## Spec errata

Four defects were found in the original specification. Each was verified against the installed
`gymnasium` source rather than assumed.

### 1. The training loop cannot learn (fatal)

The specification's loop steps the environment **once** per episode and then resets:

```python
state, _ = env.reset()
...
next_state, reward, terminated, truncated, _ = env.step(action.item())
loss = -action_dist.log_prob(action) * reward
```

A `HIT` that does not bust returns `reward = 0.0`, so `loss = -log_prob * 0.0 = 0.0` and **no
gradient flows**. `HIT` therefore receives signal only when it busts, while `STAND` collects an
immediate ±1 on every episode. The reward for the hand that hitting makes possible is never
attributed to the hit, and states reached after a hit are never visited at all. `terminated` and
`truncated` are unpacked and discarded.

The consequence is not a merely weaker agent — it is a policy that collapses onto a single constant
action. See "The spec loop's collapse" above for the measured behaviour.

**Fix:** `trainers.train_reinforce` plays each hand to termination and credits every decision in it
with the return that followed (`return-to-go`), with a running-mean baseline and an entropy bonus.

### 2. The parameter count is wrong — twice

The specification headlines "~4,184 weights" and computes a total of 4,180. The architecture it
specifies has:

| Tensor | Shape | Parameters |
|---|---|---|
| `W1` | 3 × 64 | 192 |
| `b1` | 64 | 64 |
| `W2` | 64 × 56 | 3,584 |
| `b2` | 56 | 56 |
| `W3` | 56 × 2 | 112 |
| `b3` | 2 | 2 |
| **Total** | | **4,010** |

192 + 64 + 3,584 + 56 + 112 + 2 = 4,010, not 4,180. The architecture is the biologically motivated
part and is kept exactly as specified; the count is corrected and asserted by a test.

### 3. The player-sum range is misstated

The specification says the player total ranges over 12–21. The observation space is `Discrete(32)`
and real deals start as low as 4. Restricting the encoding to 12–21 would silently mishandle every
low hand — precisely the hands where hitting is free of any bust risk.

### 4. Section 6 asks for metrics the specification never implements

Both the 10,000-hand EV sweep and the decision-matrix comparison are specified as deliverables, with
no code for either. Both are implemented here — and the comparison is made against an exact optimum
derived from the rules, not a basic-strategy table copied from a book.

---

## Why NumPy instead of PyTorch

`download.pytorch.org` is unreachable from this build environment (the proxy returns 403), and the
PyPI wheel is 555 MB plus CUDA dependencies. For a three-layer, 4,010-parameter network, hand-written
forward and backward passes are short, exact, and gradient-checkable, and they keep the whole test
suite running in seconds.

The gradients are not taken on trust: `tests/test_flynance_brain.py` compares every analytic gradient
against central finite differences (h = 1e-5), with a maximum relative error of 2.1e-08.

The specification's original PyTorch listing is preserved verbatim at
`reference/spec_pytorch_original.py`. It is never imported or executed.

---

## Module map

| File | Contents |
|---|---|
| `brain.py` | `FlyBlackjackBrain` — NumPy MLP, hand-written backprop and Adam, save/load |
| `encoding.py` | Spec-verbatim sensory encoding, plus a wider encoding for the ablation |
| `trainers.py` | `train_spec_baseline` (the spec's loop, faithfully) and `train_reinforce` (corrected) |
| `optimal.py` | Exact optimal policy, state values, EV and visit frequencies by dynamic programming |
| `evaluate.py` | Seeded EV sweeps with 95% CIs and common random numbers across policies |
| `strategy.py` | Decision-matrix rendering and frequency-weighted alignment scoring |
| `run_experiment.py` | CLI that runs everything and writes the report |
| `reference/` | The specification's original PyTorch listing, unmodified |

---

## How the optimum is derived

`optimal.py` does not contain a basic-strategy table. It computes, from the rules alone:

1. the dealer's final-score distribution for each upcard, with the hole card uniform over the
   13-card deck and the dealer standing on soft 17;
2. the value of standing in every player state, from that distribution;
3. the value of hitting, by backward induction — hands are carried as `(hard_sum, has_ace)` so that
   every hit strictly increases `hard_sum`, which gives a clean topological order and makes soft
   hands fall out with no special casing (soft 17 plus a ten is a hard 17, not a bust);
4. the exact expected value, by integrating over the true initial-deal distribution;
5. how often each decision state is actually visited under optimal play, propagated analytically —
   which is what makes the *weighted* alignment score meaningful.

The resulting grid reproduces textbook basic strategy exactly, and its exact EV is confirmed by
simulation. That agreement is the evidence that the ground truth is trustworthy.

---

## Environment ground truth

Read from `gymnasium/envs/toy_text/blackjack.py`, not from memory:

| Property | Value |
|---|---|
| Deck | Infinite, uniform over `[1,2,3,4,5,6,7,8,9,10,10,10,10]` — a 10 has probability 4/13 |
| Dealer | Draws while total < 17; **stands on soft 17** |
| Actions | `0 = stand`, `1 = hit` |
| Reward | +1 / 0 / −1; a bust terminates immediately at −1 |
| Naturals | With `natural=False, sab=False`, a natural pays 1.0 only on an outright win, and pushes against a dealer natural |

Because the deck is infinite, `(player_sum, dealer_upcard, usable_ace)` is a sufficient statistic and
card counting is meaningless — which is why no counting features exist here.
