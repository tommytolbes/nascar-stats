"""Decision matrices and agreement scoring against the DP optimum.

Two jobs:

``decision_matrix`` / ``render_matrix``
    Turn a :data:`~flynance.evaluate.PolicyFn` into the familiar basic-strategy
    grid (rows = player total, columns = dealer upcard) and print it, optionally
    diffed against a reference grid.

``alignment`` / ``render_alignment``
    Score a policy against an ``optimal.OptimalSolution``: what fraction of its
    decisions match the exact dynamic-programming optimum, both unweighted and
    weighted by how often each state is actually visited, plus a ranked list of
    the mistakes that cost the most expected value.

Nothing in this module imports :mod:`flynance.optimal`; an ``OptimalSolution``
is always passed in, and is only required to expose ``policy`` and
``state_frequency`` (``action_values`` is used when available -- see
:func:`alignment`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HARD_SUMS",
    "SOFT_SUMS",
    "UPCARDS",
    "MATRIX_SEED",
    "ACTION_LETTERS",
    "Disagreement",
    "decision_matrix",
    "render_matrix",
    "alignment",
    "render_alignment",
]

#: Player totals with no usable ace.  Gymnasium can deal a hard 4 (two deuces),
#: so the grid starts at 4 -- not at 12 as the original specification claimed.
HARD_SUMS: Tuple[int, ...] = tuple(range(4, 22))

#: Player totals holding a usable ace.  A usable ace forces the total to 12..21.
SOFT_SUMS: Tuple[int, ...] = tuple(range(12, 22))

#: Dealer upcards, 1..10, where 1 is an ace (rendered as "A").
UPCARDS: Tuple[int, ...] = tuple(range(1, 11))

#: Fixed seed for the rng handed to the policy while building a matrix, so the
#: rendered grid is reproducible run to run.
MATRIX_SEED = 0

#: Action code -> cell letter.  0 = stand, 1 = hit.
ACTION_LETTERS = {0: "S", 1: "H"}


def decision_matrix(policy_fn, seed: int = MATRIX_SEED) -> Dict[str, Any]:
    """Tabulate ``policy_fn``'s action for every reachable decision state.

    Returns
    -------
    dict
        ``{'hard': grid, 'soft': grid, 'hard_sums': ..., 'soft_sums': ...,
        'upcards': ...}``.  Each grid is an ``int`` :class:`numpy.ndarray` whose
        rows are player totals (``HARD_SUMS`` = 4..21 for ``'hard'``,
        ``SOFT_SUMS`` = 12..21 for ``'soft'``) and whose columns are dealer
        upcards ``UPCARDS`` = 1..10 (column 0 is the ace).  Cell values are the
        raw actions ``0`` (stand) / ``1`` (hit); :func:`render_matrix` turns them
        into ``S``/``H``.

    Sampling note
    -------------
    A :data:`~flynance.evaluate.PolicyFn` receives an rng and is allowed to be
    **stochastic**.  This function calls it exactly once per cell with a rng
    seeded from ``seed``, so what you get back is *one sample* of a stochastic
    policy's behaviour -- reproducible, but not "the" policy.  Callers who want
    the deterministic decision (which is what a strategy chart is supposed to
    show, and what :func:`alignment` scores) should pass a greedy policy, e.g.
    ``flynance.evaluate.greedy_policy_fn(brain)``.
    """
    rng = np.random.default_rng(seed)

    def grid_for(sums: Sequence[int], usable_ace: int) -> np.ndarray:
        grid = np.empty((len(sums), len(UPCARDS)), dtype=int)
        for i, player_sum in enumerate(sums):
            for j, upcard in enumerate(UPCARDS):
                grid[i, j] = int(policy_fn((player_sum, upcard, usable_ace), rng))
        return grid

    return {
        "hard": grid_for(HARD_SUMS, 0),
        "soft": grid_for(SOFT_SUMS, 1),
        "hard_sums": HARD_SUMS,
        "soft_sums": SOFT_SUMS,
        "upcards": UPCARDS,
    }


def _upcard_labels() -> List[str]:
    return ["A" if u == 1 else str(u) for u in UPCARDS]


def _rows_for(grid: np.ndarray, rows: Optional[Sequence[int]]) -> Sequence[int]:
    if rows is not None:
        return rows
    n = grid.shape[0]
    if n == len(HARD_SUMS):
        return HARD_SUMS
    if n == len(SOFT_SUMS):
        return SOFT_SUMS
    raise ValueError(
        f"cannot infer row labels for a grid with {n} rows; pass rows=... explicitly"
    )


def _render_one(
    grid: np.ndarray,
    rows: Sequence[int],
    reference: Optional[np.ndarray],
    heading: Optional[str],
) -> Tuple[List[str], int]:
    """Render one grid; returns (lines, disagreement_count)."""
    grid = np.asarray(grid, dtype=int)
    if reference is not None:
        reference = np.asarray(reference, dtype=int)
        if reference.shape != grid.shape:
            raise ValueError(
                f"reference shape {reference.shape} does not match grid shape {grid.shape}"
            )

    labels = _upcard_labels()
    row_w = max(len("total"), max(len(str(r)) for r in rows))
    col_w = max(3, max(len(lab) for lab in labels))

    lines: List[str] = []
    if heading:
        lines.append(heading)
    lines.append(
        "total".rjust(row_w) + " | " + " ".join(lab.rjust(col_w) for lab in labels)
    )
    lines.append("-" * row_w + "-+-" + "-" * (len(labels) * (col_w + 1) - 1))

    n_disagree = 0
    for i, player_sum in enumerate(rows):
        cells = []
        for j in range(grid.shape[1]):
            letter = ACTION_LETTERS.get(int(grid[i, j]), "?")
            if reference is not None and int(grid[i, j]) != int(reference[i, j]):
                n_disagree += 1
                # Disagreeing cells are lowercased and starred, so they stand
                # out whether or not the terminal supports colour.
                letter = letter.lower() + "*"
            cells.append(letter.rjust(col_w))
        lines.append(str(player_sum).rjust(row_w) + " | " + " ".join(cells))
    return lines, n_disagree


def render_matrix(matrix, title: str, reference=None, rows: Optional[Sequence[int]] = None) -> str:
    """Render a decision matrix as an ASCII grid.

    Parameters
    ----------
    matrix:
        Either the full dict returned by :func:`decision_matrix` (both the hard
        and soft sections are rendered) or a single 2-D grid.
    title:
        Heading printed above the grid.
    reference:
        Optional grid (or full dict, matching ``matrix``) to diff against, e.g.
        the optimal strategy's matrix.  Cells that disagree are rendered
        lowercase with a trailing ``*``, and a legend plus a total disagreement
        count is appended.
    rows:
        Row labels when ``matrix`` is a bare grid whose row count is neither
        ``len(HARD_SUMS)`` nor ``len(SOFT_SUMS)``.  Normally unnecessary.

    Returns
    -------
    str
        Columns are dealer upcards labelled ``A, 2 .. 10``; rows are player
        totals.  ``H`` = hit, ``S`` = stand.
    """
    lines: List[str] = [title, "=" * len(title)]
    total_disagree = 0
    total_cells = 0

    if isinstance(matrix, Mapping):
        ref_map = reference if isinstance(reference, Mapping) else None
        if reference is not None and ref_map is None:
            raise TypeError(
                "reference must also be a decision_matrix dict when matrix is a dict"
            )
        for key, heading, default_rows in (
            ("hard", "Hard totals (no usable ace)", HARD_SUMS),
            ("soft", "Soft totals (usable ace)", SOFT_SUMS),
        ):
            if key not in matrix:
                continue
            grid = np.asarray(matrix[key], dtype=int)
            row_labels = matrix.get(f"{key}_sums", default_rows)
            ref_grid = None if ref_map is None else np.asarray(ref_map[key], dtype=int)
            section, n = _render_one(grid, row_labels, ref_grid, heading)
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(section)
            total_disagree += n
            total_cells += grid.size
    else:
        grid = np.asarray(matrix, dtype=int)
        row_labels = _rows_for(grid, rows)
        ref_grid = None if reference is None else np.asarray(reference, dtype=int)
        section, n = _render_one(grid, row_labels, ref_grid, None)
        lines.append("")
        lines.extend(section)
        total_disagree += n
        total_cells += grid.size

    lines.append("")
    lines.append("Legend: H = hit, S = stand; columns are the dealer upcard (A = ace).")
    if reference is not None:
        lines.append(
            "        lowercase + '*' marks a cell that disagrees with the reference."
        )
        lines.append(
            f"Disagreements: {total_disagree} / {total_cells} cells "
            f"({100.0 * total_disagree / total_cells:.1f}%)"
            if total_cells
            else "Disagreements: 0 / 0 cells"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Agreement with the exact optimum
# --------------------------------------------------------------------------


class Disagreement(NamedTuple):
    """One state where the policy differs from the optimum.

    A :class:`~typing.NamedTuple`, so it unpacks positionally in exactly the
    order the design contract documents --
    ``(state, agent_action, optimal_action, frequency, ev_cost)`` -- while still
    supporting attribute access and ``._asdict()`` for reporting.
    """

    state: Tuple[int, int, int]
    agent_action: int
    optimal_action: int
    frequency: float
    ev_cost: float

    @property
    def cost_weight(self) -> float:
        """``frequency * ev_cost`` -- the sort key: how much this mistake costs."""
        return self.frequency * self.ev_cost


def _ev_cost(optimal: Any, state: Tuple[int, int, int], agent_action: int,
             optimal_action: int) -> float:
    """EV lost by taking ``agent_action`` instead of ``optimal_action``.

    Uses ``optimal.action_values(state)`` -- a per-action value lookup, either a
    mapping ``{action: value}`` or a length-2 sequence indexed by action -- when
    the optimal solution provides one.  That accessor is optional in the frozen
    contract, so if it is missing (or raises for this state) we fall back to
    ``0.0``: the disagreement is still reported, it just carries no cost
    estimate and therefore sorts last.
    """
    getter = getattr(optimal, "action_values", None)
    if getter is None:
        return 0.0
    try:
        values = getter(state)
        cost = float(values[optimal_action]) - float(values[agent_action])
    except Exception:
        return 0.0
    # Numerical noise can make an "optimal" action look infinitesimally worse.
    return max(0.0, cost)


def alignment(policy_fn, optimal: Any, seed: int = MATRIX_SEED) -> Dict[str, Any]:
    """Score ``policy_fn`` against an exact ``OptimalSolution``.

    The scored state set is exactly ``optimal.policy``'s keys -- the decision
    states the dynamic program actually solved.

    Returns
    -------
    dict
        ``{'raw': float, 'weighted': float, 'disagreements': [Disagreement, ...],
        'n_states': int}`` where

        ``raw``
            Unweighted fraction of scored states whose action matches.
        ``weighted``
            The same fraction with each state weighted by
            ``optimal.state_frequency``, normalised over the states actually
            scored (so it is a true fraction even if the frequency table covers
            states the policy table does not, or vice versa).  ``0.0`` if the
            total weight is zero.
        ``disagreements``
            Sorted by ``frequency * ev_cost`` descending, so the most expensive
            mistakes come first.

    As in :func:`decision_matrix`, the policy is sampled once per state with a
    rng seeded from ``seed``; pass a greedy policy fn for a deterministic score.
    """
    rng = np.random.default_rng(seed)
    freqs = getattr(optimal, "state_frequency", {}) or {}

    n_states = 0
    n_match = 0
    weight_total = 0.0
    weight_match = 0.0
    disagreements: List[Disagreement] = []

    for state, optimal_action in optimal.policy.items():
        state = tuple(state)
        optimal_action = int(optimal_action)
        agent_action = int(policy_fn(state, rng))
        freq = float(freqs.get(state, 0.0))

        n_states += 1
        weight_total += freq
        if agent_action == optimal_action:
            n_match += 1
            weight_match += freq
        else:
            disagreements.append(
                Disagreement(
                    state=state,
                    agent_action=agent_action,
                    optimal_action=optimal_action,
                    frequency=freq,
                    ev_cost=_ev_cost(optimal, state, agent_action, optimal_action),
                )
            )

    disagreements.sort(key=lambda d: d.frequency * d.ev_cost, reverse=True)
    return {
        "raw": (n_match / n_states) if n_states else 0.0,
        "weighted": (weight_match / weight_total) if weight_total > 0 else 0.0,
        "disagreements": disagreements,
        "n_states": n_states,
    }


def render_alignment(result: Mapping[str, Any], top_n: int = 10) -> str:
    """Summarise :func:`alignment` output: agreement percentages + worst mistakes."""
    disagreements = list(result.get("disagreements", []))
    n_states = int(result.get("n_states", 0))
    lines = [
        "Alignment with DP optimum",
        "=========================",
        f"Raw agreement:      {100.0 * float(result.get('raw', 0.0)):6.2f}%"
        + (f"  ({n_states - len(disagreements)}/{n_states} states)" if n_states else ""),
        f"Weighted agreement: {100.0 * float(result.get('weighted', 0.0)):6.2f}%"
        "  (states weighted by visit frequency under optimal play)",
        f"Disagreements:      {len(disagreements)}",
    ]

    if not disagreements:
        lines.append("")
        lines.append("No disagreements -- the policy matches the optimum everywhere.")
        return "\n".join(lines)

    shown = disagreements[: max(0, top_n)]
    lines.append("")
    lines.append(f"Top {len(shown)} most costly disagreements (frequency x EV cost):")
    header = ("player", "dealer", "ace", "agent", "optimal", "freq", "EV cost", "freq*cost")
    rows = []
    for d in shown:
        player_sum, upcard, usable_ace = d.state
        rows.append(
            (
                str(player_sum),
                "A" if upcard == 1 else str(upcard),
                "yes" if usable_ace else "no",
                ACTION_LETTERS.get(d.agent_action, "?"),
                ACTION_LETTERS.get(d.optimal_action, "?"),
                f"{d.frequency:.5f}",
                f"{d.ev_cost:.4f}",
                f"{d.frequency * d.ev_cost:.6f}",
            )
        )
    widths = [len(h) for h in header]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))
    lines.append("  ".join(h.rjust(widths[j]) for j, h in enumerate(header)))
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(cell.rjust(widths[j]) for j, cell in enumerate(row)))
    if len(disagreements) > len(shown):
        lines.append(f"... and {len(disagreements) - len(shown)} more")
    return "\n".join(lines)
