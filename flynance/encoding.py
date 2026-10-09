"""Sensory encoding: raw Blackjack observations -> fly-brain input vectors.

A *state* is always the raw gymnasium observation tuple
``(player_sum, dealer_upcard, usable_ace)`` where ``dealer_upcard`` is ``1..10``
(1 is an ace) and ``usable_ace`` is ``0``/``1``.

Two encodings live here:

``preprocess_state``
    The specification's verbatim mapping, and the default everywhere in this
    project.  Three scalars, shape ``(3,)``.

``preprocess_state_rich``
    A wider encoding kept only for the documented ablation (see the "Risks"
    section of the plan).  Nothing in the default pipeline uses it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "INPUT_SIZE",
    "RICH_INPUT_SIZE",
    "preprocess_state",
    "encode_batch",
    "preprocess_state_rich",
    "encode_batch_rich",
]

#: Input width of the spec encoding -- matches ``FlyBlackjackBrain.LAYER_SIZES[0]``.
INPUT_SIZE = 3

#: Input width of the ablation encoding: 1 (player sum) + 10 (dealer one-hot) + 1 (ace).
RICH_INPUT_SIZE = 12
FULL_INPUT_SIZE = 29


def preprocess_state(state: tuple[int, int, int]) -> np.ndarray:
    """Encode one observation exactly as the original specification does.

    ``(player_sum, dealer_upcard, usable_ace)`` ->
    ``[player_sum / 21.0, dealer_upcard / 10.0, float(usable_ace)]``

    Parameters
    ----------
    state:
        Raw environment observation tuple.

    Returns
    -------
    np.ndarray
        Shape ``(3,)``, dtype ``float64``.
    """
    player_sum, dealer_upcard, usable_ace = state
    return np.array(
        [player_sum / 21.0, dealer_upcard / 10.0, float(usable_ace)],
        dtype=np.float64,
    )


def encode_batch(states: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """Encode a sequence of observations with :func:`preprocess_state`.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)``, dtype ``float64``.  ``N`` may be zero.
    """
    if len(states) == 0:
        return np.zeros((0, INPUT_SIZE), dtype=np.float64)
    return np.stack([preprocess_state(s) for s in states]).astype(np.float64, copy=False)


def preprocess_state_rich(state: tuple[int, int, int]) -> np.ndarray:
    """Ablation encoding: one-hot dealer upcard instead of a single scalar.

    Layout (width :data:`RICH_INPUT_SIZE` = 12)::

        [0]      player_sum / 21.0
        [1:11]   one-hot over dealer upcard 1..10  (index 0 == ace)
        [11]     float(usable_ace)

    Why the spec encoding is suspected to be limiting
    -------------------------------------------------
    The spec maps the dealer upcard to ``dealer_upcard / 10.0``, a single
    monotone scalar.  A dealer **ace** has value 1 and therefore lands at
    ``0.1`` -- numerically the *weakest* upcard on that axis, immediately below
    a dealer 2 at ``0.2``.  Strategically the ace is the dealer's *strongest*
    upcard: it is the card the player must play most defensively against,
    roughly on par with a dealer 10.  Optimal play as a function of this scalar
    is therefore non-monotonic -- aggressive at ``0.1``, relaxed through the
    middle (``0.2``-``0.6``), aggressive again at ``0.7``-``1.0``.

    Fitting a non-monotonic decision boundary from one input scalar forces the
    tiny network to spend hidden units carving that axis into pieces, capacity
    it could otherwise spend on the player-sum/soft-hand interaction.  The
    one-hot version removes the false ordering entirely: each upcard gets its
    own free parameter and the ace is no longer pinned next to the 2.  Whether
    this actually matters at 4,010 parameters is exactly what the ablation
    measures; it is *not* enabled by default, because the point of the project
    is to reproduce and correct the specification, not to redesign it.

    Returns
    -------
    np.ndarray
        Shape ``(RICH_INPUT_SIZE,)``, dtype ``float64``.
    """
    player_sum, dealer_upcard, usable_ace = state
    out = np.zeros(RICH_INPUT_SIZE, dtype=np.float64)
    out[0] = player_sum / 21.0
    upcard = int(dealer_upcard)
    if 1 <= upcard <= 10:
        out[upcard] = 1.0  # out[1] == ace ... out[10] == ten
    out[11] = float(usable_ace)
    return out


def encode_batch_rich(states: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """Batch form of :func:`preprocess_state_rich`; shape ``(N, RICH_INPUT_SIZE)``."""
    if len(states) == 0:
        return np.zeros((0, RICH_INPUT_SIZE), dtype=np.float64)
    return np.stack([preprocess_state_rich(s) for s in states]).astype(np.float64, copy=False)


def preprocess_state_full_onehot(state: tuple[int, int, int]) -> np.ndarray:
    """Every feature one-hot: no false ordering anywhere in the input.

    Layout (width :data:`FULL_INPUT_SIZE` = 29)::

        [0:18]   one-hot over player sum 4..21
        [18:28]  one-hot over dealer upcard 1..10  (index 18 == ace)
        [28]     float(usable_ace)

    :func:`preprocess_state_rich` fixes the dealer upcard but still feeds the
    player's total as a single scalar ``sum / 21``, which quietly asserts that 16
    is "close to" 17. For hit/stand that is false in the way that matters most:
    16 and 17 sit on opposite sides of the sharpest boundary in the game, while
    12 and 13 are near-interchangeable. One-hot removes that assumption too, at
    the cost of a wider input layer.

    Returns
    -------
    np.ndarray
        Shape ``(FULL_INPUT_SIZE,)``, dtype ``float64``.
    """
    player_sum, dealer_upcard, usable_ace = state
    out = np.zeros(FULL_INPUT_SIZE, dtype=np.float64)
    if 4 <= player_sum <= 21:
        out[player_sum - 4] = 1.0
    upcard = int(dealer_upcard)
    if 1 <= upcard <= 10:
        out[17 + upcard] = 1.0   # out[18] == ace ... out[27] == ten
    out[28] = float(usable_ace)
    return out


def encode_batch_full_onehot(states: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """Batch form of :func:`preprocess_state_full_onehot`; shape ``(N, 29)``."""
    if len(states) == 0:
        return np.zeros((0, FULL_INPUT_SIZE), dtype=np.float64)
    return np.stack([preprocess_state_full_onehot(s) for s in states]).astype(np.float64, copy=False)
