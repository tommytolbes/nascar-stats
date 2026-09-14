"""Measurement: play real Blackjack hands and report expected value with error bars.

Everything here talks to the *real* environment --
``gymnasium.make("Blackjack-v1", natural=False, sab=False)`` -- never to a model
of it.  See "Environment ground truth" in
``docs/superpowers/specs/2026-09-14-flynance-blackjack-design.md``:

* actions are ``0 = stand``, ``1 = hit``;
* a state is the raw observation tuple ``(player_sum, dealer_upcard, usable_ace)``
  with ``dealer_upcard`` in ``1..10`` (1 is an ace);
* reward is ``+1`` win / ``0`` push / ``-1`` loss, so the mean reward per hand is
  directly the expected value of one unit wagered.

The headline entry points are :func:`evaluate` (one policy) and :func:`compare`
(several policies over a *common* sequence of deals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np

__all__ = [
    "PolicyFn",
    "EvalResult",
    "MAX_STEPS_PER_HAND",
    "evaluate",
    "compare",
    "render_comparison",
    "always_stand",
    "dealer_mimic",
    "greedy_policy_fn",
]

#: A policy maps ``(state, rng) -> action``.  ``state`` is the raw environment
#: observation tuple ``(player_sum, dealer_upcard, usable_ace)``; ``rng`` is a
#: :class:`numpy.random.Generator` the policy may use for stochastic choices (a
#: deterministic policy simply ignores it).  The return value is ``0`` (stand)
#: or ``1`` (hit).
PolicyFn = Callable[[Tuple[int, int, int], np.random.Generator], int]

#: Safety net for a pathological policy.  Against gymnasium's infinite deck,
#: hitting forever always busts eventually (every card is worth at least 1), so
#: a hand cannot legitimately last this long; exceeding the cap means the policy
#: or the environment is broken, and :func:`evaluate` raises rather than hangs.
MAX_STEPS_PER_HAND = 50


@dataclass
class EvalResult:
    """Summary statistics for one policy over ``n_hands`` completed hands.

    Attributes
    ----------
    ev:
        Mean reward per hand (units wagered won per hand).
    stderr:
        Sample standard deviation of per-hand rewards divided by ``sqrt(n)``.
    ci95:
        ``(ev - 1.96 * stderr, ev + 1.96 * stderr)``.
    win_rate, loss_rate, push_rate:
        Fractions of hands with reward > 0, < 0 and == 0.  Sum to 1.0.
    n_hands:
        Number of hands played.
    label:
        Human-readable policy name, used by :func:`render_comparison`.
    """

    ev: float
    stderr: float
    ci95: Tuple[float, float]
    win_rate: float
    loss_rate: float
    push_rate: float
    n_hands: int
    label: str


def _make_env() -> gym.Env:
    """The one environment configuration this project measures against."""
    return gym.make("Blackjack-v1", natural=False, sab=False)


def _hand_seeds(seed: int, n_hands: int) -> np.ndarray:
    """Deterministically derive one environment seed per hand from ``seed``.

    Uses :class:`numpy.random.SeedSequence`, so the seeds are well spread and a
    pure function of ``(seed, n_hands)``.  Hand ``i`` always gets the same seed
    for a given base ``seed``, which is what makes the common-random-numbers
    guarantee in :func:`compare` hold.
    """
    return np.random.SeedSequence(seed).generate_state(n_hands, dtype=np.uint32)


def evaluate(
    policy_fn: PolicyFn,
    n_hands: int = 10_000,
    seed: int = 0,
    label: str = "",
) -> EvalResult:
    """Play ``n_hands`` complete Blackjack hands under ``policy_fn``.

    Each hand starts from ``env.reset(seed=hand_seed)`` where ``hand_seed`` comes
    from :func:`_hand_seeds`, so the sequence of deals depends only on ``seed``
    and ``n_hands`` -- never on the policy.  The policy gets its own
    ``numpy.random.Generator`` seeded from ``seed`` as well, so a stochastic
    policy is reproducible too.

    Raises
    ------
    ValueError
        If ``n_hands`` is not positive.
    RuntimeError
        If a single hand exceeds :data:`MAX_STEPS_PER_HAND` environment steps,
        which against an infinite deck can only mean a broken policy/action.
    """
    if n_hands <= 0:
        raise ValueError(f"n_hands must be positive, got {n_hands}")

    env = _make_env()
    action_rng = np.random.default_rng(seed)
    seeds = _hand_seeds(seed, n_hands)
    rewards = np.empty(n_hands, dtype=np.float64)

    try:
        for i in range(n_hands):
            state, _ = env.reset(seed=int(seeds[i]))
            total = 0.0
            for step in range(MAX_STEPS_PER_HAND + 1):
                if step == MAX_STEPS_PER_HAND:
                    raise RuntimeError(
                        f"policy {label or policy_fn!r} did not terminate hand {i} "
                        f"within {MAX_STEPS_PER_HAND} steps (last state {state!r}); "
                        "a hitting policy must bust eventually, so this indicates a bug"
                    )
                action = int(policy_fn(tuple(state), action_rng))
                if action not in (0, 1):
                    raise ValueError(
                        f"policy {label or policy_fn!r} returned invalid action "
                        f"{action!r} for state {state!r}; expected 0 (stand) or 1 (hit)"
                    )
                state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            rewards[i] = total
    finally:
        env.close()

    ev = float(rewards.mean())
    # Sample standard deviation (ddof=1); undefined for a single hand.
    sd = float(rewards.std(ddof=1)) if n_hands > 1 else 0.0
    stderr = sd / math.sqrt(n_hands)
    half = 1.96 * stderr
    return EvalResult(
        ev=ev,
        stderr=stderr,
        ci95=(ev - half, ev + half),
        win_rate=float((rewards > 0).mean()),
        loss_rate=float((rewards < 0).mean()),
        push_rate=float((rewards == 0).mean()),
        n_hands=n_hands,
        label=label,
    )


def compare(
    policies: Dict[str, PolicyFn],
    n_hands: int = 10_000,
    seed: int = 0,
) -> List[EvalResult]:
    """Evaluate several policies over the **same** sequence of deals.

    Common random numbers
    ---------------------
    Blackjack EV differences between sensible policies are on the order of a few
    percent of a unit, while the per-hand standard deviation is close to 1.0.
    Comparing policies on independent deals therefore buries the signal in deal
    variance.  This function removes that variance:

    * hand ``i`` is started with ``env.reset(seed=s_i)`` where ``s_i`` is derived
      deterministically from the base ``seed`` via :func:`_hand_seeds` -- so every
      policy is dealt an identical initial player hand and dealer upcard on hand
      ``i``;
    * every policy is handed its own ``numpy.random.Generator``, all seeded
      identically from ``seed``, so stochastic policies are reproducible and are
      not accidentally coupled to one another;
    * consequently two policies that happen to choose the same actions produce
      *bit-for-bit identical* results, and two that differ only diverge from the
      first decision on which they actually disagree.

    (Cards drawn *after* a policy's first differing action necessarily differ --
    that is the game branching, not sampling noise.  The deal itself, which is
    the dominant variance term, is shared.)

    Returns
    -------
    list[EvalResult]
        One result per policy, sorted by ``ev`` descending (best first).  Each
        result's ``label`` is the dict key.
    """
    results = [
        evaluate(policy_fn, n_hands=n_hands, seed=seed, label=label)
        for label, policy_fn in policies.items()
    ]
    results.sort(key=lambda r: r.ev, reverse=True)
    return results


def render_comparison(results: Sequence[EvalResult]) -> str:
    """Render evaluation results as an aligned ASCII table, best EV first."""
    header = ("policy", "EV/hand", "+/- 95% CI", "95% CI", "win%", "loss%", "push%", "hands")
    rows: List[Tuple[str, ...]] = []
    for r in sorted(results, key=lambda x: x.ev, reverse=True):
        half = 1.96 * r.stderr
        rows.append(
            (
                r.label or "(unlabelled)",
                f"{r.ev:+.4f}",
                f"{half:.4f}",
                f"[{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}]",
                f"{100.0 * r.win_rate:.2f}",
                f"{100.0 * r.loss_rate:.2f}",
                f"{100.0 * r.push_rate:.2f}",
                f"{r.n_hands}",
            )
        )

    widths = [len(h) for h in header]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        out = [cells[0].ljust(widths[0])]
        out += [cells[j].rjust(widths[j]) for j in range(1, len(cells))]
        return "  ".join(out).rstrip()

    lines = [fmt(header), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    if not rows:
        lines.append("(no results)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reference policies
# --------------------------------------------------------------------------


def always_stand(state: Tuple[int, int, int], rng: np.random.Generator) -> int:
    """Never take a card.  The trivial baseline any learner must beat."""
    return 0


def dealer_mimic(state: Tuple[int, int, int], rng: np.random.Generator) -> int:
    """Play the dealer's own rule: hit while ``player_sum < 17``, else stand.

    Note this mimics the *hard* dealer rule on the player's total; the dealer in
    gymnasium stands on soft 17, and so does this policy, since ``player_sum`` is
    already the soft-adjusted total.
    """
    player_sum = state[0]
    return 1 if player_sum < 17 else 0


def greedy_policy_fn(brain, encoder=None) -> PolicyFn:
    """Wrap a :class:`~flynance.brain.FlyBlackjackBrain` as a deterministic policy.

    The returned :data:`PolicyFn` takes ``argmax`` over
    ``brain.policy(encoded_state[None, :])[0]`` -- i.e. the greedy action under
    the brain's action distribution -- and ignores the ``rng`` entirely, so it is
    deterministic.

    Parameters
    ----------
    brain:
        Anything exposing ``policy(x: (B, D)) -> (B, 2)``.
    encoder:
        ``state -> (D,) ndarray``.  ``None`` (the default) means
        :func:`flynance.encoding.preprocess_state`, which is imported lazily
        inside this function so that importing :mod:`flynance.evaluate` never
        requires :mod:`flynance.encoding`.
    """
    if encoder is None:
        from flynance.encoding import preprocess_state  # lazy: see docstring

        encoder = preprocess_state

    def policy(state: Tuple[int, int, int], rng: np.random.Generator) -> int:
        x = np.asarray(encoder(tuple(state)), dtype=np.float64)
        probs = brain.policy(x[None, :])[0]
        return int(np.argmax(probs))

    return policy
