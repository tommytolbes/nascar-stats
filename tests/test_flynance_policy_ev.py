"""Tests for exact policy evaluation and the full one-hot encoding.

``optimal.policy_expected_value`` exists because sampling cannot separate a good
blackjack policy from a perfect one: over 10,000 hands the standard error is
about 0.01, while the entire gap between the two is around 0.005. These tests
pin the exact evaluator against the two things that can check it -- the closed
form it must reproduce, and a simulation it must agree with.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path so we can import flynance (same bootstrap as test_model.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance import evaluate as ev
from flynance import optimal as opt
from flynance.encoding import (FULL_INPUT_SIZE, encode_batch_full_onehot,
                               preprocess_state_full_onehot)


@pytest.fixture(scope="module")
def solution():
    return opt.solve()


# --------------------------------------------------------------------------
# Exact policy evaluation
# --------------------------------------------------------------------------

def test_optimal_policy_evaluates_to_the_optimal_value(solution):
    """The tightest check available: it must reproduce solve() exactly."""
    assert opt.policy_expected_value(solution.policy) == pytest.approx(
        solution.expected_value, abs=1e-12
    )


def test_no_policy_beats_the_optimum(solution):
    """Optimal means optimal: nothing scores above it, by construction."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        random_policy = {s: int(rng.integers(0, 2)) for s in solution.policy}
        assert opt.policy_expected_value(random_policy) <= solution.expected_value + 1e-12


def test_matches_simulation_for_reference_policies(solution):
    """Agrees with actually playing the hands, within sampling error."""
    always_stand = {s: 0 for s in solution.policy}
    mimic = {s: (1 if s[0] < 17 else 0) for s in solution.policy}

    for name, policy, policy_fn in [
        ("always_stand", always_stand, ev.always_stand),
        ("dealer_mimic", mimic, ev.dealer_mimic),
    ]:
        exact = opt.policy_expected_value(policy)
        sampled = ev.evaluate(policy_fn, n_hands=100_000, seed=11)
        assert abs(exact - sampled.ev) < 2 * sampled.stderr, (
            f"{name}: exact {exact:+.5f} vs sampled {sampled.ev:+.5f} "
            f"+/- {sampled.stderr:.5f}"
        )


def test_known_reference_values(solution):
    """Structural sanity: standing always is far worse than mimicking the dealer."""
    always_stand = opt.policy_expected_value({s: 0 for s in solution.policy})
    mimic = opt.policy_expected_value(
        {s: (1 if s[0] < 17 else 0) for s in solution.policy})

    assert always_stand == pytest.approx(-0.1864, abs=1e-3)
    assert mimic == pytest.approx(-0.0793, abs=1e-3)
    assert always_stand < mimic < solution.expected_value


def test_missing_states_fall_back_to_the_dealer_rule(solution):
    """A partial policy must not crash; uncovered states use hit-below-17."""
    partial = {s: a for s, a in solution.policy.items() if s[0] >= 12}
    value = opt.policy_expected_value(partial)
    assert -1.0 < value < 0.0


def test_one_wrong_cell_costs_expected_value(solution):
    """Flipping the single most expensive decision must lower the exact EV."""
    worst = (15, 1, 0)   # hard 15 vs a dealer ace: the costliest cell in the grid
    damaged = dict(solution.policy)
    damaged[worst] = 1 - damaged[worst]
    assert opt.policy_expected_value(damaged) < solution.expected_value


# --------------------------------------------------------------------------
# Full one-hot encoding
# --------------------------------------------------------------------------

def test_full_onehot_layout():
    x = preprocess_state_full_onehot((16, 1, 0))
    assert x.shape == (FULL_INPUT_SIZE,)
    assert x[16 - 4] == 1.0          # player sum 16
    assert x[18] == 1.0              # dealer ace
    assert x[28] == 0.0              # no usable ace
    assert x.sum() == 2.0            # exactly two hot bits


def test_full_onehot_separates_every_upcard_and_total():
    """No two distinct states share an encoding -- that is the whole point."""
    seen = {}
    for player_sum in range(4, 22):
        for upcard in range(1, 11):
            for ace in (0, 1):
                key = preprocess_state_full_onehot((player_sum, upcard, ace)).tobytes()
                assert key not in seen, f"{(player_sum, upcard, ace)} collides with {seen[key]}"
                seen[key] = (player_sum, upcard, ace)


def test_full_onehot_ace_is_not_adjacent_to_a_two():
    """The flaw being removed: under the spec encoding an ace sits next to a 2."""
    ace = preprocess_state_full_onehot((16, 1, 0))
    two = preprocess_state_full_onehot((16, 2, 0))
    ten = preprocess_state_full_onehot((16, 10, 0))
    # Every distinct upcard is now exactly the same distance from every other.
    assert np.linalg.norm(ace - two) == pytest.approx(np.linalg.norm(ace - ten))


def test_full_onehot_batch_matches_single():
    states = [(12, 3, 0), (20, 10, 1), (4, 1, 0)]
    batch = encode_batch_full_onehot(states)
    assert batch.shape == (3, FULL_INPUT_SIZE)
    for i, s in enumerate(states):
        assert np.array_equal(batch[i], preprocess_state_full_onehot(s))
