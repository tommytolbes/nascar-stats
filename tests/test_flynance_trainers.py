"""Tests for the two training loops.

The central claims under test are the ones the project exists to demonstrate:
the specification's single-step loop throws away the learning signal for HIT
(S4), and both trainers are exactly reproducible from a seed (S7).
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path so we can import flynance (same bootstrap as test_model.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance.brain import FlyBlackjackBrain
from flynance.trainers import (
    HITTABLE_SUMS,
    _policy_gradient_dlogits,
    _softmax,
    stand_rate_on_hittable,
    train_reinforce,
    train_spec_baseline,
)


# --------------------------------------------------------------------------
# The policy-gradient rule itself
# --------------------------------------------------------------------------

def test_dlogits_is_zero_when_reward_is_zero():
    """The spec's bug, isolated.

    A HIT that does not bust returns reward 0.0, so the spec's
    ``loss = -log_prob * reward`` has an identically zero gradient: the
    experience is silently discarded.
    """
    probs = np.array([[0.3, 0.7]])
    dlogits = _policy_gradient_dlogits(
        probs, np.array([1]), np.array([0.0]), entropy_coef=0.0
    )
    assert np.allclose(dlogits, 0.0)


def test_dlogits_sign_follows_reward():
    """Positive reward pushes probability toward the taken action, negative away."""
    probs = np.array([[0.5, 0.5]])
    reward_up = _policy_gradient_dlogits(
        probs, np.array([1]), np.array([1.0]), entropy_coef=0.0
    )
    reward_down = _policy_gradient_dlogits(
        probs, np.array([1]), np.array([-1.0]), entropy_coef=0.0
    )
    # Gradient descent moves logits by -grad, so a rewarded action needs a
    # negative gradient on its own logit.
    assert reward_up[0, 1] < 0
    assert reward_down[0, 1] > 0
    assert np.allclose(reward_up, -reward_down)


def test_dlogits_matches_finite_differences_with_entropy():
    """Analytic gradient of the full loss (REINFORCE + entropy bonus)."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 2))
    actions = rng.integers(0, 2, size=4)
    weights = rng.normal(size=4)
    entropy_coef = 0.05

    def loss(z):
        probs = _softmax(z)
        log_probs = np.log(probs)
        chosen = log_probs[np.arange(len(actions)), actions]
        entropy = -(probs * log_probs).sum(axis=1)
        return float(np.sum(-weights * chosen - entropy_coef * entropy))

    analytic = _policy_gradient_dlogits(
        _softmax(logits), actions, weights, entropy_coef=entropy_coef
    )

    h = 1e-6
    numerical = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up = logits.copy()
            up[i, j] += h
            down = logits.copy()
            down[i, j] -= h
            numerical[i, j] = (loss(up) - loss(down)) / (2 * h)

    assert np.max(np.abs(analytic - numerical)) < 1e-7


def test_entropy_bonus_pushes_toward_uniform():
    """With no reward signal, the entropy term alone flattens a peaked policy."""
    probs = np.array([[0.99, 0.01]])
    dlogits = _policy_gradient_dlogits(
        probs, np.array([0]), np.array([0.0]), entropy_coef=0.1
    )
    # Descending this gradient lowers the dominant logit and raises the other.
    assert dlogits[0, 0] > 0
    assert dlogits[0, 1] < 0


# --------------------------------------------------------------------------
# stand_rate_on_hittable
# --------------------------------------------------------------------------

class _ConstantBrain:
    """Stub exposing just the ``policy`` method the diagnostic needs."""

    def __init__(self, action):
        self.action = action

    def policy(self, x):
        probs = np.zeros((len(x), 2))
        probs[:, self.action] = 1.0
        return probs


def test_stand_rate_extremes():
    assert stand_rate_on_hittable(_ConstantBrain(0)) == 1.0
    assert stand_rate_on_hittable(_ConstantBrain(1)) == 0.0


def test_hittable_sums_cover_the_decision_range():
    assert min(HITTABLE_SUMS) == 4
    assert max(HITTABLE_SUMS) == 16


# --------------------------------------------------------------------------
# S4: the spec loop degenerates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed,expected_mode", [(0, "hit"), (4, "stand")])
def test_spec_baseline_collapses_to_a_constant_action(seed, expected_mode):
    """S4: the specification's loop collapses onto one action everywhere.

    A non-busting HIT pays 0 and produces no gradient, so HIT is only ever
    reinforced by its busts, while STAND collects an immediate +/-1 every
    episode. With no baseline to calibrate two negative signals against each
    other, the policy falls into whichever corner the seed sends it -- seed 0
    hits every hand (and therefore busts every hand), seed 4 stands on
    everything.
    """
    brain, history = train_spec_baseline(episodes=20_000, seed=seed)
    assert len(history.episode_rewards) == 20_000

    stand_rate = stand_rate_on_hittable(brain)
    assert max(stand_rate, 1.0 - stand_rate) > 0.9, (
        f"expected a degenerate policy, got stand_rate={stand_rate:.3f}"
    )
    if expected_mode == "stand":
        assert stand_rate > 0.9
    else:
        assert stand_rate < 0.1


def test_spec_baseline_always_hit_mode_loses_every_hand():
    """The always-hit collapse is total: hitting forever busts with probability 1."""
    from flynance import evaluate as ev

    brain, _ = train_spec_baseline(episodes=20_000, seed=0)
    assert stand_rate_on_hittable(brain) < 0.1
    result = ev.evaluate(ev.greedy_policy_fn(brain), n_hands=2_000, seed=5)
    assert result.ev == pytest.approx(-1.0)
    assert result.win_rate == 0.0


def test_spec_baseline_records_config():
    _, history = train_spec_baseline(episodes=200, seed=0)
    assert history.config["trainer"] == "spec_baseline"
    assert history.config["episodes"] == 200


# --------------------------------------------------------------------------
# REINFORCE mechanics
# --------------------------------------------------------------------------

def test_reinforce_runs_full_hands_and_records_outcomes():
    _, history = train_reinforce(episodes=500, seed=5)
    assert len(history.episode_rewards) == 500
    # Terminal reward of a hand under natural=False is always -1, 0 or +1.
    assert set(history.episode_rewards) <= {-1.0, 0.0, 1.0}


def test_reinforce_learns_to_hit_low_totals():
    """The corrected loop must escape the degenerate corners the spec falls into.

    Optimal play hits every one of these states, so a competent policy stands on
    few of them. This is the cheap smoke test; the full EV and alignment numbers
    come from ``run_experiment.py``.
    """
    brain, _ = train_reinforce(episodes=50_000, seed=5)
    stand_rate = stand_rate_on_hittable(brain)
    assert stand_rate < 0.5, f"still standing on {stand_rate:.1%} of hittable states"


def test_reinforce_beats_the_spec_loop_on_ev():
    """The headline claim, at small scale: fixing credit assignment pays."""
    from flynance import evaluate as ev

    fly, _ = train_reinforce(episodes=50_000, seed=5)
    spec, _ = train_spec_baseline(episodes=50_000, seed=5)

    results = ev.compare(
        {
            "reinforce": ev.greedy_policy_fn(fly),
            "spec_baseline": ev.greedy_policy_fn(spec),
        },
        n_hands=20_000,
        seed=1,
    )
    by_label = {r.label: r for r in results}
    fly_result = by_label["reinforce"]
    spec_result = by_label["spec_baseline"]
    assert fly_result.ci95[0] > spec_result.ci95[1], (
        f"reinforce {fly_result.ev:+.4f} did not clearly beat "
        f"spec_baseline {spec_result.ev:+.4f}"
    )


def test_reinforce_batching_steps_less_often():
    """With batch_episodes=N the optimizer steps once per N hands."""
    steps = {"count": 0}
    original_step = FlyBlackjackBrain.step

    def counting_step(self):
        steps["count"] += 1
        original_step(self)

    FlyBlackjackBrain.step = counting_step
    try:
        steps["count"] = 0
        train_reinforce(episodes=100, seed=1, batch_episodes=10)
        assert steps["count"] == 10
    finally:
        FlyBlackjackBrain.step = original_step


def test_reinforce_rejects_mismatched_encoder_width():
    """A wider encoder without a matching brain must fail loudly, not silently."""
    from flynance.encoding import preprocess_state_rich

    with pytest.raises(ValueError, match="features but the brain expects"):
        train_reinforce(episodes=10, seed=0, encoder=preprocess_state_rich)


def test_reinforce_accepts_a_wider_brain_for_the_ablation():
    """The encoding ablation needs a brain whose input layer matches the encoder."""
    from flynance.encoding import RICH_INPUT_SIZE, preprocess_state_rich

    class WideBrain(FlyBlackjackBrain):
        LAYER_SIZES = (RICH_INPUT_SIZE, 64, 56, 2)

    wide = WideBrain(rng=np.random.default_rng(0))
    trained, history = train_reinforce(
        episodes=200, seed=0, encoder=preprocess_state_rich, brain=wide
    )
    assert trained is wide
    assert len(history.episode_rewards) == 200


def test_moving_average_blocks():
    _, history = train_reinforce(episodes=400, seed=2)
    averages = history.moving_average(window=100)
    assert len(averages) == 4
    assert all(-1.0 <= a <= 1.0 for a in averages)


# --------------------------------------------------------------------------
# S7: reproducibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("trainer", [train_spec_baseline, train_reinforce])
def test_same_seed_is_bit_identical(trainer):
    """S7: a seed fully determines the run."""
    brain_a, history_a = trainer(episodes=1_000, seed=11)
    brain_b, history_b = trainer(episodes=1_000, seed=11)

    assert history_a.episode_rewards == history_b.episode_rewards
    for name in brain_a.params:
        assert np.array_equal(brain_a.params[name], brain_b.params[name])


@pytest.mark.parametrize("trainer", [train_spec_baseline, train_reinforce])
def test_different_seeds_diverge(trainer):
    brain_a, _ = trainer(episodes=1_000, seed=11)
    brain_b, _ = trainer(episodes=1_000, seed=12)
    assert not np.array_equal(brain_a.params["W1"], brain_b.params["W1"])


def test_checkpoints_are_recorded():
    _, history = train_reinforce(episodes=1_000, seed=4, checkpoint_every=250)
    assert len(history.checkpoints) == 4
    assert history.checkpoints[-1]["episode"] == 1_000
    for checkpoint in history.checkpoints:
        assert 0.0 <= checkpoint["stand_rate_on_hittable"] <= 1.0
