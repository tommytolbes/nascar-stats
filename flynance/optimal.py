"""Exact dynamic-programming solution of Gymnasium's ``Blackjack-v1``.

Everything in this module is derived from the rules implemented by
``gymnasium/envs/toy_text/blackjack.py`` for ``natural=False, sab=False``.
Nothing is copied from a printed basic-strategy chart and nothing is estimated
by simulation.

Rules modelled
--------------
* Infinite deck: every draw is uniform over ``[1,2,...,9,10,10,10,10]``, so
  ``P(10) = 4/13`` and ``P(c) = 1/13`` for ``c`` in ``1..9``.
* An ace counts as 11 whenever that keeps the total at or below 21 (a *usable
  ace*), otherwise it counts as 1.
* The dealer draws while their total is below 17 and **stands on soft 17**.
* Player actions are ``0 = stand`` and ``1 = hit``; busting terminates the hand
  immediately with reward ``-1``.
* Standing pays ``+1 / 0 / -1`` by comparing the player's total against the
  dealer's, with a busted dealer scoring 0.
* With ``natural=False, sab=False`` a natural is *not* special-cased: it is just
  a 21 that can stand, and it pushes against a dealer 21.

State encoding
--------------
A *state* is always the raw environment observation
``(player_sum, dealer_upcard, usable_ace)`` where ``player_sum`` is the
effective total (ace already counted as 11 when usable), ``dealer_upcard`` is
``1..10`` (1 is an ace) and ``usable_ace`` is ``0`` or ``1``.

Internally a hand is carried as ``(hard_sum, has_ace)`` where ``hard_sum``
counts every ace as 1.  The effective total is ``hard_sum + 10`` when an ace is
present and that stays at or below 21, otherwise ``hard_sum``.  Every hit
strictly increases ``hard_sum``, which gives the natural topological order for
the backward induction and for the forward frequency propagation.  This also
makes soft hands fall out correctly: soft 17 is ``hard_sum = 7``, so drawing a
ten yields ``hard_sum = 17`` -- a *hard 17*, not a bust.

Public interface
----------------
``dealer_outcome_distribution(upcard)``
    Memoized mapping ``final dealer score -> probability`` for a dealer showing
    ``upcard``; score ``0`` means the dealer busted.  The hole card is uniform
    over the 13-card deck (no conditioning of any kind).  The probabilities sum
    to 1 to within floating-point noise.

``solve()``
    Returns an :class:`OptimalSolution` with the optimal policy, the exact
    state values, the exact expected value of optimal play from the initial
    deal, and the analytic visit frequency of every decision state.

``action_values(state)``
    Returns ``(stand_value, hit_value)`` for a single state, so other modules
    can price a disagreement with the optimal policy: the EV cost of taking
    action ``a`` in ``state`` is ``max(values) - values[a]``.  Backed by a
    cached solve, so repeated calls are cheap.

``optimal_action(state)``
    Convenience wrapper returning ``0`` or ``1`` for a state, usable directly as
    a policy function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

__all__ = [
    "DECK",
    "CARD_PROBABILITY",
    "DEALER_STANDS_AT",
    "OptimalSolution",
    "dealer_outcome_distribution",
    "solve",
    "action_values",
    "optimal_action",
    "hand_total",
]

#: The infinite deck, exactly as gymnasium defines it.
DECK = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)

#: ``card value -> probability`` for a single draw: 4/13 for a ten, 1/13 otherwise.
CARD_PROBABILITY = {
    card: DECK.count(card) / len(DECK) for card in sorted(set(DECK))
}

#: The dealer draws while their total is strictly below this, and stands on soft 17.
DEALER_STANDS_AT = 17

#: All dealer upcards, 1 (ace) through 10.
UPCARDS = tuple(range(1, 11))


def hand_total(hard_sum: int, has_ace: bool) -> tuple[int, int]:
    """Return ``(effective_total, usable_ace)`` for a hand.

    ``hard_sum`` counts every ace as 1.  An ace is *usable* when promoting it to
    11 keeps the total at or below 21.  A busted hand is returned unchanged with
    ``usable_ace == 0``.
    """
    if has_ace and hard_sum + 10 <= 21:
        return hard_sum + 10, 1
    return hard_sum, 0


def _draw(hard_sum: int, has_ace: bool, card: int) -> tuple[int, bool]:
    """Apply one drawn ``card`` to ``(hard_sum, has_ace)``."""
    return hard_sum + card, has_ace or card == 1


# --------------------------------------------------------------------------
# Dealer
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _dealer_from_hand(hard_sum: int, has_ace: bool) -> tuple[tuple[int, float], ...]:
    """Distribution over final dealer scores from a partial dealer hand.

    Returned as a sorted tuple of ``(score, probability)`` pairs so it can be
    memoized safely.  A score of ``0`` means the dealer busted.
    """
    if hard_sum > 21:
        return ((0, 1.0),)

    total, _ = hand_total(hard_sum, has_ace)
    if total >= DEALER_STANDS_AT:
        # Stands on soft 17 as well: the check is on the effective total.
        return ((total, 1.0),)

    accumulated: dict[int, float] = {}
    for card, probability in CARD_PROBABILITY.items():
        next_sum, next_ace = _draw(hard_sum, has_ace, card)
        for score, sub_probability in _dealer_from_hand(next_sum, next_ace):
            accumulated[score] = accumulated.get(score, 0.0) + probability * sub_probability
    return tuple(sorted(accumulated.items()))


@lru_cache(maxsize=None)
def _dealer_outcome_pairs(upcard: int) -> tuple[tuple[int, float], ...]:
    """Memoized dealer outcome distribution, keyed by upcard."""
    if upcard not in UPCARDS:
        raise ValueError(f"dealer upcard must be 1..10, got {upcard!r}")

    accumulated: dict[int, float] = {}
    for hole, probability in CARD_PROBABILITY.items():
        hard_sum, has_ace = _draw(upcard, upcard == 1, hole)
        for score, sub_probability in _dealer_from_hand(hard_sum, has_ace):
            accumulated[score] = accumulated.get(score, 0.0) + probability * sub_probability
    return tuple(sorted(accumulated.items()))


def dealer_outcome_distribution(upcard: int) -> dict[int, float]:
    """Return ``{final dealer score: probability}`` for a dealer showing ``upcard``.

    A score of ``0`` means the dealer busted.  The hole card is uniform over the
    13-card deck; the distribution is *not* conditioned on anything the player
    holds (the deck is infinite, so the cards are independent anyway).  The
    probabilities sum to 1 within 1e-12.

    The computation is memoized; a fresh dict is returned on each call so
    callers may mutate the result freely.
    """
    return dict(_dealer_outcome_pairs(int(upcard)))


# --------------------------------------------------------------------------
# Player
# --------------------------------------------------------------------------


def _stand_value(player_total: int, upcard: int) -> float:
    """EV of standing on ``player_total`` against ``upcard``.

    Reward is ``cmp(player_score, dealer_score)`` with a busted dealer scoring 0,
    exactly as the environment computes it.
    """
    value = 0.0
    for dealer_score, probability in _dealer_outcome_pairs(upcard):
        if dealer_score == 0 or player_total > dealer_score:
            value += probability
        elif player_total < dealer_score:
            value -= probability
    return value


def _states_by_hard_sum() -> dict[int, list[tuple[int, int]]]:
    """Group every reachable player hand by its underlying ``hard_sum``.

    Each entry maps ``hard_sum -> [(player_sum, usable_ace), ...]``.  Hard hands
    run 4..21 (two cards minimum); soft hands run 12..21 (an ace plus 1..10),
    whose hard sums are 2..11.  Because every hit strictly increases
    ``hard_sum``, iterating this mapping in descending key order is a valid
    backward-induction order and ascending order is a valid forward order.
    """
    grouped: dict[int, list[tuple[int, int]]] = {}
    for player_sum in range(4, 22):  # hard hands
        grouped.setdefault(player_sum, []).append((player_sum, 0))
    for player_sum in range(12, 22):  # soft hands; hard_sum = player_sum - 10
        grouped.setdefault(player_sum - 10, []).append((player_sum, 1))
    return grouped


@dataclass
class OptimalSolution:
    """The exact solution of the game.

    Attributes
    ----------
    policy:
        ``state -> 0 (stand) | 1 (hit)``, the optimal action.
    state_value:
        ``state -> float``, the expected reward from that state under optimal
        play.
    expected_value:
        The exact expected reward of optimal play from the initial deal,
        integrated over the true deal distribution (two player cards and one
        dealer upcard, each uniform over the 13-card deck).
    state_frequency:
        ``state -> float``, how often each decision state is *visited* under
        optimal play, normalized to sum to 1.  Computed analytically by pushing
        the initial-deal probability mass forward through the optimal policy.
    stand_value / hit_value:
        Per-state action values, kept so a disagreement with the optimal policy
        can be priced.  See :func:`action_values`.
    """

    policy: dict[tuple[int, int, int], int]
    state_value: dict[tuple[int, int, int], float]
    expected_value: float
    state_frequency: dict[tuple[int, int, int], float]
    stand_value: dict[tuple[int, int, int], float] = field(default_factory=dict)
    hit_value: dict[tuple[int, int, int], float] = field(default_factory=dict)

    def action_values(self, state: tuple[int, int, int]) -> tuple[float, float]:
        """Return ``(stand_value, hit_value)`` for ``state``."""
        key = (int(state[0]), int(state[1]), int(state[2]))
        return self.stand_value[key], self.hit_value[key]

    def action_value(self, state: tuple[int, int, int], action: int) -> float:
        """Return the EV of taking ``action`` in ``state`` and playing optimally after."""
        return self.action_values(state)[int(action)]

    def ev_cost(self, state: tuple[int, int, int], action: int) -> float:
        """Return how much EV taking ``action`` in ``state`` gives up (>= 0)."""
        values = self.action_values(state)
        return max(values) - values[int(action)]


def _initial_deal_distribution() -> dict[tuple[int, int, int], float]:
    """Exact distribution of the observation after the deal.

    Two player cards and one dealer upcard, each drawn independently from the
    13-card deck.  A dealt 21 is just a standing 21 under ``natural=False,
    sab=False``.
    """
    deal: dict[tuple[int, int, int], float] = {}
    for first, p_first in CARD_PROBABILITY.items():
        for second, p_second in CARD_PROBABILITY.items():
            hard_sum, has_ace = _draw(first, first == 1, second)
            player_sum, usable = hand_total(hard_sum, has_ace)
            for upcard, p_up in CARD_PROBABILITY.items():
                state = (player_sum, upcard, usable)
                deal[state] = deal.get(state, 0.0) + p_first * p_second * p_up
    return deal


def solve() -> OptimalSolution:
    """Solve the game exactly by backward induction.

    Returns a freshly built :class:`OptimalSolution`; the caller owns the
    returned dicts.  Use :func:`action_values` when a cached solve is enough.
    """
    grouped = _states_by_hard_sum()

    policy: dict[tuple[int, int, int], int] = {}
    state_value: dict[tuple[int, int, int], float] = {}
    stand_values: dict[tuple[int, int, int], float] = {}
    hit_values: dict[tuple[int, int, int], float] = {}

    # Backward induction: every hit strictly increases hard_sum, so descending
    # hard_sum order guarantees successors are already solved.
    for hard_sum in sorted(grouped, reverse=True):
        for player_sum, usable in grouped[hard_sum]:
            has_ace = bool(usable)
            for upcard in UPCARDS:
                state = (player_sum, upcard, usable)

                stand = _stand_value(player_sum, upcard)

                hit = 0.0
                for card, probability in CARD_PROBABILITY.items():
                    next_sum, next_ace = _draw(hard_sum, has_ace, card)
                    if next_sum > 21:
                        hit -= probability  # bust: immediate -1
                        continue
                    next_total, next_usable = hand_total(next_sum, next_ace)
                    hit += probability * state_value[(next_total, upcard, next_usable)]

                stand_values[state] = stand
                hit_values[state] = hit
                if hit > stand:
                    policy[state] = 1
                    state_value[state] = hit
                else:
                    policy[state] = 0
                    state_value[state] = stand

    deal = _initial_deal_distribution()
    expected_value = sum(prob * state_value[state] for state, prob in deal.items())

    # Forward pass: push the deal's probability mass through the optimal policy.
    # Ascending hard_sum order guarantees a state's inflow is complete before it
    # is propagated, because every hit strictly increases hard_sum.
    visits: dict[tuple[int, int, int], float] = {state: 0.0 for state in state_value}
    for state, probability in deal.items():
        visits[state] += probability

    for hard_sum in sorted(grouped):
        for player_sum, usable in grouped[hard_sum]:
            has_ace = bool(usable)
            for upcard in UPCARDS:
                state = (player_sum, upcard, usable)
                mass = visits[state]
                if mass == 0.0 or policy[state] == 0:
                    continue
                for card, probability in CARD_PROBABILITY.items():
                    next_sum, next_ace = _draw(hard_sum, has_ace, card)
                    if next_sum > 21:
                        continue  # busted hands are terminal, not decision states
                    next_total, next_usable = hand_total(next_sum, next_ace)
                    visits[(next_total, upcard, next_usable)] += mass * probability

    total_visits = sum(visits.values())
    state_frequency = {state: mass / total_visits for state, mass in visits.items()}

    return OptimalSolution(
        policy=policy,
        state_value=state_value,
        expected_value=expected_value,
        state_frequency=state_frequency,
        stand_value=stand_values,
        hit_value=hit_values,
    )


@lru_cache(maxsize=1)
def _cached_solution() -> OptimalSolution:
    return solve()


def action_values(state: tuple[int, int, int]) -> tuple[float, float]:
    """Return ``(stand_value, hit_value)`` for ``state`` from a cached solve.

    ``stand_value`` is the EV of standing (computed from the dealer outcome
    distribution) and ``hit_value`` is the EV of taking one card and then
    playing optimally.  The EV cost of preferring action ``a`` over the optimal
    action is ``max(values) - values[a]``.
    """
    return _cached_solution().action_values(state)


def optimal_action(state: tuple[int, int, int]) -> int:
    """Return the optimal action (``0`` stand, ``1`` hit) for ``state``."""
    key = (int(state[0]), int(state[1]), int(state[2]))
    return _cached_solution().policy[key]

def policy_expected_value(policy: dict[tuple[int, int, int], int]) -> float:
    """Exact expected value of an arbitrary hit/stand policy, by backward induction.

    The Monte-Carlo sweep in :mod:`flynance.evaluate` measures a policy by
    playing it; over 10,000 hands its standard error is about 0.01, which is
    larger than the entire gap between a good policy and a perfect one. This
    computes the same quantity exactly, so two policies can be separated without
    sampling noise at all.

    Same machinery as :func:`solve`, with one change: instead of taking the
    better of stand and hit at each state, it takes whichever action ``policy``
    names. States the policy does not cover fall back to the dealer's own rule
    (hit below 17), matching how the agents are evaluated elsewhere.

    Parameters
    ----------
    policy:
        Maps ``(player_sum, dealer_upcard, usable_ace)`` to 0 (stand) or 1 (hit).

    Returns
    -------
    float
        Expected value per hand, integrated over the true initial deal.
    """
    grouped = _states_by_hard_sum()
    value: dict[tuple[int, int, int], float] = {}

    # Every hit strictly increases hard_sum, so descending order guarantees a
    # state's successors are solved before it is.
    for hard_sum in sorted(grouped, reverse=True):
        for player_sum, usable in grouped[hard_sum]:
            has_ace = bool(usable)
            for upcard in UPCARDS:
                state = (player_sum, upcard, usable)
                action = policy.get(state)
                if action is None:
                    action = 0 if player_sum >= 17 else 1

                if action == 0:
                    value[state] = _stand_value(player_sum, upcard)
                    continue

                total = 0.0
                for card, probability in CARD_PROBABILITY.items():
                    next_sum, next_ace = _draw(hard_sum, has_ace, card)
                    if next_sum > 21:
                        total -= probability  # bust
                        continue
                    next_total, next_usable = hand_total(next_sum, next_ace)
                    total += probability * value[(next_total, upcard, next_usable)]
                value[state] = total

    deal = _initial_deal_distribution()
    return sum(prob * value[state] for state, prob in deal.items())
