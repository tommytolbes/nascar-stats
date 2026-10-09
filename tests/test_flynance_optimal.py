"""Tests for the exact DP solution of Gymnasium's Blackjack-v1.

The last test in this file is the load-bearing one: it plays 200,000 seeded
hands of the real environment under the DP-optimal policy and checks that the
simulated mean reward agrees with the analytically computed expected value.
"""

import math
import os
import sys

import pytest

# Add project root to path so we can import flynance
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance import optimal  # noqa: E402
from flynance.optimal import (  # noqa: E402
    CARD_PROBABILITY,
    action_values,
    dealer_outcome_distribution,
    solve,
)

UPCARDS = tuple(range(1, 11))
HARD_TOTALS = tuple(range(4, 22))
SOFT_TOTALS = tuple(range(12, 22))

STAND = 0
HIT = 1


@pytest.fixture(scope="module")
def solution():
    return solve()


def all_states():
    for upcard in UPCARDS:
        for total in HARD_TOTALS:
            yield (total, upcard, 0)
        for total in SOFT_TOTALS:
            yield (total, upcard, 1)


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def test_card_probabilities_match_the_infinite_deck():
    assert CARD_PROBABILITY[10] == pytest.approx(4 / 13)
    for card in range(1, 10):
        assert CARD_PROBABILITY[card] == pytest.approx(1 / 13)
    assert sum(CARD_PROBABILITY.values()) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Dealer outcome distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("upcard", UPCARDS)
def test_dealer_distribution_sums_to_one(upcard):
    distribution = dealer_outcome_distribution(upcard)
    assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(p >= 0.0 for p in distribution.values())


@pytest.mark.parametrize("upcard", UPCARDS)
def test_dealer_scores_are_bust_or_a_standing_total(upcard):
    # The dealer stands on soft 17, so every non-bust outcome is 17..21.
    for score in dealer_outcome_distribution(upcard):
        assert score == 0 or 17 <= score <= 21


def test_dealer_distribution_is_memoized_but_returns_a_fresh_dict():
    first = dealer_outcome_distribution(6)
    first[0] = 123.0
    second = dealer_outcome_distribution(6)
    assert second[0] != 123.0


def test_dealer_bust_probability_ordering():
    bust = {upcard: dealer_outcome_distribution(upcard)[0] for upcard in UPCARDS}

    # Structural fact: a dealer showing 6 busts most often, and an ace or a ten
    # busts least often.  Both hold in the computed distribution.
    assert max(bust, key=bust.get) == 6
    assert min(bust, key=bust.get) == 1
    assert bust[1] < bust[10] < min(bust[u] for u in (2, 3, 4, 5, 6, 7, 8, 9))

    # The 2..6 "bust cards" all bust more than the 7..10/ace "strong cards".
    assert min(bust[u] for u in (2, 3, 4, 5, 6)) > max(bust[u] for u in (7, 8, 9, 10, 1))

    # Bust probability rises monotonically from 2 through 6 and falls
    # monotonically from 7 through 10.
    assert bust[2] < bust[3] < bust[4] < bust[5] < bust[6]
    assert bust[7] > bust[8] > bust[9] > bust[10]


def test_dealer_ace_upcard_reaches_21_often():
    # A dealer ace makes exactly 21 whenever the hole card is a ten (4/13),
    # which is why standing on 21 against an ace is worth far less than 0.9.
    assert dealer_outcome_distribution(1)[21] > 4 / 13


# ---------------------------------------------------------------------------
# Policy structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total", [17, 18, 19, 20, 21])
@pytest.mark.parametrize("upcard", UPCARDS)
def test_stand_on_every_hard_17_through_21(solution, total, upcard):
    assert solution.policy[(total, upcard, 0)] == STAND


@pytest.mark.parametrize("upcard", [2, 3])
def test_hit_hard_12_against_dealer_2_and_3(solution, upcard):
    assert solution.policy[(12, upcard, 0)] == HIT


@pytest.mark.parametrize("total", [13, 14, 15, 16])
@pytest.mark.parametrize("upcard", [2, 3, 4, 5, 6])
def test_stand_hard_13_through_16_against_dealer_2_through_6(solution, total, upcard):
    assert solution.policy[(total, upcard, 0)] == STAND


@pytest.mark.parametrize("total", [12, 13, 14, 15, 16])
@pytest.mark.parametrize("upcard", [7, 8, 9, 10, 1])
def test_hit_hard_12_through_16_against_dealer_7_through_ace(solution, total, upcard):
    assert solution.policy[(total, upcard, 0)] == HIT


@pytest.mark.parametrize("total", HARD_TOTALS[:8])  # hard 4..11
@pytest.mark.parametrize("upcard", UPCARDS)
def test_always_hit_hard_totals_below_12(solution, total, upcard):
    assert solution.policy[(total, upcard, 0)] == HIT


@pytest.mark.parametrize("upcard", UPCARDS)
def test_hit_soft_17_against_every_upcard(solution, upcard):
    assert solution.policy[(17, upcard, 1)] == HIT


@pytest.mark.parametrize("upcard", UPCARDS)
def test_stand_on_soft_19_through_21(solution, upcard):
    for total in (19, 20, 21):
        assert solution.policy[(total, upcard, 1)] == STAND


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_state_values_are_within_the_reward_range(solution):
    assert len(solution.state_value) == len(UPCARDS) * (len(HARD_TOTALS) + len(SOFT_TOTALS))
    for state in all_states():
        value = solution.state_value[state]
        assert -1.0 <= value <= 1.0, f"{state} -> {value}"


def test_state_value_is_the_max_of_the_two_action_values(solution):
    for state in all_states():
        stand, hit = solution.action_values(state)
        assert solution.state_value[state] == pytest.approx(max(stand, hit))
        assert solution.policy[state] == (HIT if hit > stand else STAND)
        assert solution.ev_cost(state, solution.policy[state]) == pytest.approx(0.0)


def test_module_level_action_values_match_a_fresh_solve(solution):
    for state in [(16, 10, 0), (12, 2, 0), (18, 9, 1), (21, 1, 0)]:
        assert action_values(state) == pytest.approx(solution.action_values(state))
    assert optimal.optimal_action((16, 10, 0)) == solution.policy[(16, 10, 0)]


def test_stand_value_of_hard_21_is_high(solution):
    stand_21 = {u: solution.action_values((21, u, 0))[STAND] for u in UPCARDS}

    # 21 cannot be beaten, so standing on 21 is the best stand value available
    # for every upcard.
    for upcard in UPCARDS:
        best = max(
            solution.action_values((total, upcard, 0))[STAND] for total in HARD_TOTALS
        )
        assert stand_21[upcard] == pytest.approx(best)

    assert max(stand_21.values()) > 0.9
    # Against every upcard except an ace the value clears 0.88; against an ace
    # it is far lower (~0.64) because the dealer makes 21 and pushes about 31%
    # of the time.  Asserting a blanket > 0.9 would contradict the rules.
    assert min(stand_21[u] for u in UPCARDS if u != 1) > 0.88
    assert stand_21[1] > 0.6
    weighted = sum(CARD_PROBABILITY[u] * stand_21[u] for u in UPCARDS)
    assert weighted > 0.87


def test_hitting_a_hard_21_always_busts(solution):
    for upcard in UPCARDS:
        assert solution.action_values((21, upcard, 0))[HIT] == pytest.approx(-1.0)


def test_hitting_a_soft_hand_never_busts(solution):
    # Soft 17 + ten becomes hard 17, not a bust, so the hit value can never be
    # as bad as -1 and must be strictly above the worst possible stand value.
    for total in SOFT_TOTALS:
        for upcard in UPCARDS:
            assert solution.action_values((total, upcard, 1))[HIT] > -1.0


def test_expected_value_is_a_small_negative_number(solution):
    assert -0.10 < solution.expected_value < 0.0


def test_expected_value_matches_the_deal_weighted_state_values(solution):
    # Recompute the deal integral independently of solve().
    total = 0.0
    for first, p1 in CARD_PROBABILITY.items():
        for second, p2 in CARD_PROBABILITY.items():
            hard = first + second
            usable = 1 if (first == 1 or second == 1) and hard + 10 <= 21 else 0
            player_sum = hard + 10 if usable else hard
            for upcard, p3 in CARD_PROBABILITY.items():
                total += p1 * p2 * p3 * solution.state_value[(player_sum, upcard, usable)]
    assert total == pytest.approx(solution.expected_value, abs=1e-12)


# ---------------------------------------------------------------------------
# State frequency
# ---------------------------------------------------------------------------


def test_state_frequency_is_a_probability_distribution(solution):
    assert set(solution.state_frequency) == set(solution.state_value)
    assert sum(solution.state_frequency.values()) == pytest.approx(1.0, abs=1e-12)
    for state, freq in solution.state_frequency.items():
        assert freq >= 0.0, f"{state} -> {freq}"


def test_frequent_states_are_the_ones_you_actually_see(solution):
    # Hard 20 against a ten is the single most common decision state; states the
    # optimal policy can never reach carry zero mass.
    assert solution.state_frequency[(20, 10, 0)] > 0.0
    top = max(solution.state_frequency, key=solution.state_frequency.get)
    assert top[1] == 10  # a ten is the most likely upcard (4/13)
    # Hard 4 is only reachable as 2+2 at the deal, so it is rare but non-zero.
    assert 0.0 < solution.state_frequency[(4, 10, 0)] < solution.state_frequency[(20, 10, 0)]


# ---------------------------------------------------------------------------
# Monte Carlo cross-check against the real environment
# ---------------------------------------------------------------------------


def test_monte_carlo_matches_the_exact_expected_value(solution):
    # Deliberately unmarked so it runs by default: this is success criterion S3.
    gym = pytest.importorskip("gymnasium")

    n_hands = 200_000
    seed = 20260914

    env = gym.make("Blackjack-v1", natural=False, sab=False)
    policy = solution.policy

    total = 0.0
    total_sq = 0.0
    obs, _ = env.reset(seed=seed)
    for hand in range(n_hands):
        if hand:
            obs, _ = env.reset()
        terminated = False
        reward = 0.0
        while not terminated:
            state = (int(obs[0]), int(obs[1]), int(obs[2]))
            action = policy[state]
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not truncated
        total += reward
        total_sq += reward * reward
    env.close()

    mean = total / n_hands
    variance = max(total_sq / n_hands - mean * mean, 0.0)
    stderr = math.sqrt(variance / n_hands)
    exact = solution.expected_value

    assert abs(mean - exact) <= 2.0 * stderr, (
        f"simulated EV {mean:.6f} over {n_hands} hands (stderr {stderr:.6f}) "
        f"disagrees with exact DP EV {exact:.6f} by {abs(mean - exact):.6f} "
        f"= {abs(mean - exact) / stderr:.2f} standard errors"
    )
