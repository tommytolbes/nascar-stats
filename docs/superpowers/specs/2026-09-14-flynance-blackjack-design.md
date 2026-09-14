# Agentic Flynance — Blackjack Fly-Brain Agent — Design Spec

**Date:** 2026-09-14
**Project:** Agentic Flynance (`flynance/`)
**Status:** Approved for implementation

---

## Goal

Build a working, reproducible, self-verifying implementation of the Agentic Flynance experiment that
delivers the two results the original specification asks for — a 10,000-hand expected-value sweep and
a decision-matrix comparison against optimal basic strategy — and quantifies exactly how much the
original specification's training loop costs in performance.

---

## Environment ground truth

Verified by reading `gymnasium/envs/toy_text/blackjack.py` (gymnasium 1.3.0), not from memory:

| Property | Value |
|---|---|
| Deck | Infinite, drawn uniformly from `[1,2,3,4,5,6,7,8,9,10,10,10,10]` — a 10 has probability 4/13 |
| Observation | `(player_sum, dealer_upcard, usable_ace)`, spaces `Discrete(32) x Discrete(11) x Discrete(2)` |
| Real player sums | 4–21 at deal (the spec's stated 12–21 is wrong) |
| Dealer upcard | 1–10, where 1 is an ace |
| Actions | `0 = stand`, `1 = hit` |
| Dealer policy | Draws while `sum_hand < 17`; **stands on soft 17** |
| Reward | `+1` win, `0` push, `-1` loss; a bust terminates immediately at `-1` |
| Naturals | With `natural=False, sab=False` a natural pays `1.0` only when it wins outright, and pushes against a dealer natural |

Because the deck is infinite, `(player_sum, dealer_upcard, usable_ace)` is a sufficient statistic for
the game and card counting is meaningless.

---

## Architecture

The original specification's network is preserved exactly. Only the training procedure is corrected.

```
[ Blackjack Environment ]
        |
        v
[ Sensory Encoding ]      player_sum/21.0, dealer_upcard/10.0, float(usable_ace)
        |
        v
[ Sensory Layer ]         3 -> 64   ReLU     (antennal lobe / Kenyon cell projection)
        |
        v
[ Mushroom Body ]         64 -> 56  ReLU     (mushroom body intrinsic sub-network)
        |
        v
[ Motor Layer ]           56 -> 2   logits   (0 = STAND, 1 = HIT)
        |
        v
[ Neuromodulation ]       +1 dopaminergic | 0 neutral | -1 octopaminergic
```

**Parameter count (authoritative):**

| Tensor | Shape | Parameters |
|---|---|---|
| `W1` | 3 x 64 | 192 |
| `b1` | 64 | 64 |
| `W2` | 64 x 56 | 3,584 |
| `b2` | 56 | 56 |
| `W3` | 56 x 2 | 112 |
| `b3` | 2 | 2 |
| **Total** | | **4,010** |

The specification claims 4,180 (its own addition is wrong) and headlines "~4,184". The architecture
is the biologically motivated part and is kept as-is; the count is corrected to 4,010 and asserted by
a test.

---

## Why NumPy and not PyTorch

`download.pytorch.org` is unreachable from the build environment (proxy returns 403) and the PyPI
wheel is 555 MB plus CUDA dependencies. For a 4,010-parameter, three-layer ReLU network, hand-written
forward and backward passes are short, exact, gradient-checkable, and run the full test suite in
milliseconds. The specification's PyTorch listing is preserved verbatim at
`flynance/reference/spec_pytorch_original.py` for the record.

---

## Interface contract

Frozen before implementation so components can be built independently. All arrays are
`numpy.ndarray` of dtype `float64` unless stated otherwise. A *state* is always the raw environment
tuple `(player_sum, dealer_upcard, usable_ace)`.

### `flynance/brain.py`

```python
class FlyBlackjackBrain:
    LAYER_SIZES = (3, 64, 56, 2)
    N_PARAMETERS = 4010

    def __init__(self, rng: np.random.Generator, lr: float = 0.005,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None
    def forward(self, x: np.ndarray) -> np.ndarray      # (B,3) -> (B,2) logits; caches activations
    def backward(self, dlogits: np.ndarray) -> None     # (B,2) upstream grad; accumulates into grads
    def step(self) -> None                              # Adam update, then zeroes accumulated grads
    def zero_grad(self) -> None
    def policy(self, x: np.ndarray) -> np.ndarray       # (B,3) -> (B,2) softmax probabilities
    def num_parameters(self) -> int                     # == 4010
    def save(self, path: str | Path) -> None            # .npz
    @classmethod
    def load(cls, path: str | Path) -> "FlyBlackjackBrain"
```

Initialization: He/Kaiming normal for weights (`std = sqrt(2 / fan_in)`), zeros for biases, all drawn
from the injected `rng` so runs are reproducible. `forward` must accept a batch; a single state is
passed as shape `(1, 3)`.

### `flynance/encoding.py`

```python
def preprocess_state(state: tuple[int, int, int]) -> np.ndarray   # (3,), spec-verbatim
def encode_batch(states: Sequence[tuple[int, int, int]]) -> np.ndarray   # (N,3)
def preprocess_state_rich(state: tuple[int, int, int]) -> np.ndarray     # optional ablation encoding
RICH_INPUT_SIZE: int    # input width of the rich encoding
```

`preprocess_state` is the default everywhere and is exactly the specification's mapping. The rich
encoding exists only for the documented ablation described under "Risks" in the plan.

### `flynance/optimal.py`

```python
@dataclass
class OptimalSolution:
    policy: dict[tuple[int, int, int], int]           # state -> 0 (stand) | 1 (hit)
    state_value: dict[tuple[int, int, int], float]    # state -> EV under optimal play
    expected_value: float                             # exact EV of optimal play from the deal
    state_frequency: dict[tuple[int, int, int], float]  # visit frequency under optimal play, sums to 1

def dealer_outcome_distribution(upcard: int) -> dict[int, float]   # final dealer score -> prob; 0 == bust
def solve() -> OptimalSolution
```

Exact dynamic programming over the rules above: dealer outcome distribution per upcard (hole card
uniform over the 13-card deck), then backward induction over player states including soft hands.
No simulation, no table copied from a book.

### `flynance/evaluate.py`

```python
PolicyFn = Callable[[tuple[int, int, int], np.random.Generator], int]

@dataclass
class EvalResult:
    ev: float; stderr: float; ci95: tuple[float, float]
    win_rate: float; loss_rate: float; push_rate: float
    n_hands: int; label: str

def evaluate(policy_fn: PolicyFn, n_hands: int = 10_000, seed: int = 0,
             label: str = "") -> EvalResult
def compare(policies: dict[str, PolicyFn], n_hands: int = 10_000,
            seed: int = 0) -> list[EvalResult]     # same seed => same hands for every policy
def render_comparison(results: list[EvalResult]) -> str
def always_stand(state, rng) -> int
def dealer_mimic(state, rng) -> int                # hit while player_sum < 17
def greedy_policy_fn(brain, encoder=preprocess_state) -> PolicyFn
```

`compare` must deal every policy the *same* sequence of hands (common random numbers) so differences
between policies are not swamped by deal variance.

### `flynance/strategy.py`

```python
def decision_matrix(policy_fn: PolicyFn) -> dict   # {'hard': grid, 'soft': grid}; rows=player sum, cols=dealer upcard 1..10
def render_matrix(matrix: dict, title: str, reference: dict | None = None) -> str
def alignment(policy_fn: PolicyFn, optimal: OptimalSolution) -> dict
#   {'raw': float, 'weighted': float, 'disagreements': [(state, agent_action, optimal_action, freq, ev_cost)]}
```

`raw` is the unweighted fraction of decision cells matching the optimal policy. `weighted` weights
each state by `optimal.state_frequency`. `ev_cost` is `state_value[optimal] - value_of_agent_action`,
so disagreements can be ranked by how much they actually cost.

### `flynance/trainers.py`

```python
@dataclass
class TrainingHistory:
    episode_rewards: list[float]; checkpoints: list[dict]; config: dict

def train_spec_baseline(episodes: int, seed: int, lr: float = 0.005, ...) -> tuple[FlyBlackjackBrain, TrainingHistory]
def train_reinforce(episodes: int, seed: int, lr: float = 0.005, gamma: float = 1.0,
                    baseline: bool = True, entropy_coef: float = 0.01,
                    batch_episodes: int = 1, ...) -> tuple[FlyBlackjackBrain, TrainingHistory]
```

`train_spec_baseline` reproduces the specification's loop exactly — one `env.step` per episode,
`loss = -log_prob * reward` — so its failure is measured rather than assumed. `train_reinforce` runs
full episodes with return-to-go, an optional running-mean baseline, and an entropy bonus.

---

## Success criteria

| # | Criterion | Checked by |
|---|---|---|
| S1 | Brain has exactly 4,010 parameters | `tests/test_flynance_brain.py` |
| S2 | Analytic gradients match finite differences to < 1e-6 | `tests/test_flynance_brain.py` |
| S3 | DP optimum shows known structure (stand on hard 17+, hit hard 12 vs dealer 2 and 3) and its exact EV lies within 2 standard errors of a 200,000-hand simulation of the same policy | `tests/test_flynance_optimal.py` |
| S4 | `spec_baseline` degenerates: stand-rate on hittable states > 0.9 | `tests/test_flynance_trainers.py` |
| S5 | `reinforce` beats always-stand and `spec_baseline` with non-overlapping 95% CIs over 10,000 hands | `run_experiment.py` report |
| S6 | `reinforce` reaches >= 85% frequency-weighted agreement with the DP optimum | `strategy.alignment` |
| S7 | Same seed produces identical metrics across runs | `tests/test_flynance_trainers.py` |

---

## Non-goals

Splits, doubling down, insurance, card counting (meaningless against an infinite deck), GPU support,
changes to any existing NASCAR code in this repository, and open-ended hyperparameter search.
