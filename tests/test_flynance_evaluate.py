"""Tests for flynance.evaluate and flynance.strategy.

Deliberately self-contained: every policy used here is a stub defined in this
file, so these tests pass without brain.py / encoding.py / optimal.py existing.
The one test that touches another module guards the import with
``pytest.importorskip``.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path so we can import flynance
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance import evaluate as ev
from flynance import strategy as st


# --------------------------------------------------------------------------
# Stub policies
# --------------------------------------------------------------------------


def stub_stand(state, rng):
    """Secretly identical to evaluate.always_stand, written independently."""
    return 0


def stub_hit_forever(state, rng):
    """Always hit.  Against an infinite deck this always busts eventually."""
    return 1


def stub_hit_below_15(state, rng):
    return 1 if state[0] < 15 else 0


def stub_coinflip(state, rng):
    return int(rng.integers(0, 2))


def stub_bad_action(state, rng):
    return 7


# --------------------------------------------------------------------------
# evaluate()
# --------------------------------------------------------------------------


def test_always_stand_is_sane():
    r = ev.evaluate(ev.always_stand, n_hands=4000, seed=11, label="always_stand")

    assert r.n_hands == 4000
    assert r.label == "always_stand"
    # Standing on everything is bad but not catastrophic: the dealer busts
    # often enough that the player still takes ~38% of hands.  Measured over
    # 200,000 hands this policy sits at EV = -0.1845 +/- 0.0042.
    assert -0.30 < r.ev < -0.08
    assert r.win_rate + r.loss_rate + r.push_rate == pytest.approx(1.0)
    assert r.stderr > 0.0
    assert r.ci95[0] < r.ev < r.ci95[1]
    assert r.ci95 == pytest.approx((r.ev - 1.96 * r.stderr, r.ev + 1.96 * r.stderr))


def test_hit_forever_terminates_and_always_loses():
    # Hitting until bust can only ever end at -1, and must do so well inside
    # the per-hand step cap.
    r = ev.evaluate(stub_hit_forever, n_hands=500, seed=3)
    assert r.ev == -1.0
    assert r.loss_rate == 1.0
    assert r.win_rate == 0.0 and r.push_rate == 0.0


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        ev.evaluate(stub_bad_action, n_hands=5, seed=0)


def test_non_positive_n_hands_raises():
    with pytest.raises(ValueError):
        ev.evaluate(ev.always_stand, n_hands=0, seed=0)


def test_step_cap_is_enforced():
    # Simulate a pathological environment/policy pair by shrinking the cap to
    # something a hitting policy can exceed.
    original = ev.MAX_STEPS_PER_HAND
    ev.MAX_STEPS_PER_HAND = 1
    try:
        with pytest.raises(RuntimeError, match="within 1 steps"):
            ev.evaluate(stub_hit_forever, n_hands=200, seed=0)
    finally:
        ev.MAX_STEPS_PER_HAND = original


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_results():
    a = ev.evaluate(ev.dealer_mimic, n_hands=1000, seed=42, label="x")
    b = ev.evaluate(ev.dealer_mimic, n_hands=1000, seed=42, label="x")
    assert a == b


def test_different_seed_gives_different_results():
    a = ev.evaluate(ev.dealer_mimic, n_hands=1000, seed=42)
    b = ev.evaluate(ev.dealer_mimic, n_hands=1000, seed=43)
    assert a.ev != b.ev


def test_stochastic_policy_is_reproducible():
    a = ev.evaluate(stub_coinflip, n_hands=800, seed=7)
    b = ev.evaluate(stub_coinflip, n_hands=800, seed=7)
    assert a == b


# --------------------------------------------------------------------------
# compare() -- common random numbers.  The most important test in the file.
# --------------------------------------------------------------------------


def test_compare_uses_common_random_numbers():
    """Two secretly-identical policies must produce bit-identical results.

    If every policy did not face the same dealt hands, these two would differ by
    sampling noise.  Exact equality is the proof that common random numbers are
    in force.
    """
    results = ev.compare(
        {
            "always_stand": ev.always_stand,
            "stub_stand": stub_stand,
            "lambda_stand": lambda s, rng: 0,
        },
        n_hands=3000,
        seed=5,
    )
    assert len(results) == 3
    evs = {r.label: r.ev for r in results}
    assert evs["always_stand"] == evs["stub_stand"] == evs["lambda_stand"]

    by_label = {r.label: r for r in results}
    for field in ("stderr", "ci95", "win_rate", "loss_rate", "push_rate"):
        a = getattr(by_label["always_stand"], field)
        b = getattr(by_label["lambda_stand"], field)
        assert a == b


def test_compare_matches_standalone_evaluate():
    """compare() must be the same measurement as evaluate() at the same seed."""
    solo = ev.evaluate(ev.dealer_mimic, n_hands=1500, seed=9, label="dealer_mimic")
    both = ev.compare(
        {"dealer_mimic": ev.dealer_mimic, "always_stand": ev.always_stand},
        n_hands=1500,
        seed=9,
    )
    got = {r.label: r for r in both}["dealer_mimic"]
    assert got == solo


def test_compare_is_sorted_best_first():
    results = ev.compare(
        {
            "hit_forever": stub_hit_forever,
            "always_stand": ev.always_stand,
            "dealer_mimic": ev.dealer_mimic,
        },
        n_hands=2000,
        seed=1,
    )
    assert [r.ev for r in results] == sorted((r.ev for r in results), reverse=True)
    assert results[-1].label == "hit_forever"


def test_dealer_mimic_beats_always_stand():
    results = ev.compare(
        {"always_stand": ev.always_stand, "dealer_mimic": ev.dealer_mimic},
        n_hands=20000,
        seed=2026,
    )
    by_label = {r.label: r for r in results}
    stand = by_label["always_stand"]
    mimic = by_label["dealer_mimic"]
    assert mimic.ev > stand.ev
    # And by a margin comfortably larger than the noise in the difference.
    pooled = (stand.stderr ** 2 + mimic.stderr ** 2) ** 0.5
    assert mimic.ev - stand.ev > 3.0 * pooled


# --------------------------------------------------------------------------
# render_comparison
# --------------------------------------------------------------------------


def test_render_comparison_contains_headers_and_labels():
    results = ev.compare(
        {"always_stand": ev.always_stand, "dealer_mimic": ev.dealer_mimic},
        n_hands=400,
        seed=0,
    )
    text = ev.render_comparison(results)
    assert text
    for token in ("policy", "EV/hand", "95% CI", "win%", "loss%", "push%", "hands"):
        assert token in text
    assert "always_stand" in text and "dealer_mimic" in text
    # Header + rule + one row per policy.
    assert len(text.splitlines()) == 4


def test_render_comparison_sorts_best_first():
    results = [
        ev.EvalResult(-0.4, 0.01, (-0.42, -0.38), 0.4, 0.5, 0.1, 100, "worse"),
        ev.EvalResult(-0.1, 0.01, (-0.12, -0.08), 0.45, 0.45, 0.1, 100, "better"),
    ]
    lines = ev.render_comparison(results).splitlines()
    assert "better" in lines[2] and "worse" in lines[3]


# --------------------------------------------------------------------------
# greedy_policy_fn (no torch, no brain module needed -- duck-typed stub)
# --------------------------------------------------------------------------


class _StubBrain:
    """Minimal ``brain.policy``-compatible object: always prefers ``action``."""

    def __init__(self, action):
        self.action = action
        self.seen = []

    def policy(self, x):
        assert x.ndim == 2 and x.shape[0] == 1
        self.seen.append(x.copy())
        probs = np.zeros((1, 2))
        probs[0, self.action] = 0.9
        probs[0, 1 - self.action] = 0.1
        return probs


def test_greedy_policy_fn_takes_argmax():
    rng = np.random.default_rng(0)
    for action in (0, 1):
        brain = _StubBrain(action)
        fn = ev.greedy_policy_fn(brain, encoder=lambda s: np.asarray(s, dtype=float))
        assert fn((16, 7, 0), rng) == action
        assert brain.seen[-1].shape == (1, 3)


def test_greedy_policy_fn_default_encoder_uses_encoding_module():
    pytest.importorskip("flynance.encoding")
    from flynance.encoding import preprocess_state

    brain = _StubBrain(1)
    fn = ev.greedy_policy_fn(brain)
    assert fn((16, 7, 0), np.random.default_rng(0)) == 1
    np.testing.assert_allclose(brain.seen[-1][0], preprocess_state((16, 7, 0)))


def test_greedy_policy_fn_is_evaluable():
    brain = _StubBrain(0)  # equivalent to always_stand
    fn = ev.greedy_policy_fn(brain, encoder=lambda s: np.asarray(s, dtype=float))
    greedy = ev.evaluate(fn, n_hands=500, seed=4)
    stand = ev.evaluate(ev.always_stand, n_hands=500, seed=4)
    assert greedy.ev == stand.ev


# --------------------------------------------------------------------------
# strategy.decision_matrix
# --------------------------------------------------------------------------


def test_decision_matrix_dimensions():
    m = st.decision_matrix(stub_stand)
    assert m["hard"].shape == (18, 10)  # hard totals 4..21
    assert m["soft"].shape == (10, 10)  # soft totals 12..21
    assert tuple(m["hard_sums"]) == tuple(range(4, 22))
    assert tuple(m["soft_sums"]) == tuple(range(12, 22))
    assert tuple(m["upcards"]) == tuple(range(1, 11))


def _cell_letters(text):
    """Every cell letter in a rendered matrix (grid rows are the '<sum> |' lines)."""
    letters = []
    for line in text.splitlines():
        head, sep, body = line.partition("|")
        if sep and head.strip().isdigit():
            letters.extend(body.split())
    return letters


def test_decision_matrix_constant_stand_is_all_stand():
    m = st.decision_matrix(stub_stand)
    assert (m["hard"] == 0).all()
    assert (m["soft"] == 0).all()
    cells = _cell_letters(st.render_matrix(m, "stand"))
    assert len(cells) == 280
    assert set(cells) == {"S"}


def test_decision_matrix_constant_hit_is_all_hit():
    m = st.decision_matrix(stub_hit_forever)
    assert (m["hard"] == 1).all()
    assert (m["soft"] == 1).all()
    assert set(_cell_letters(st.render_matrix(m, "hit"))) == {"H"}


def test_decision_matrix_passes_correct_states():
    seen = []

    def recorder(state, rng):
        seen.append(state)
        return 0

    st.decision_matrix(recorder)
    assert len(seen) == 18 * 10 + 10 * 10
    hard = [s for s in seen if s[2] == 0]
    soft = [s for s in seen if s[2] == 1]
    assert {s[0] for s in hard} == set(range(4, 22))
    assert {s[0] for s in soft} == set(range(12, 22))
    assert {s[1] for s in seen} == set(range(1, 11))


def test_decision_matrix_is_reproducible_for_stochastic_policies():
    a = st.decision_matrix(stub_coinflip)
    b = st.decision_matrix(stub_coinflip)
    np.testing.assert_array_equal(a["hard"], b["hard"])
    np.testing.assert_array_equal(a["soft"], b["soft"])
    # And it really is sampling, not collapsing to one action.
    assert set(np.unique(a["hard"])) == {0, 1}


def test_decision_matrix_matches_threshold_policy():
    m = st.decision_matrix(stub_hit_below_15)
    for i, player_sum in enumerate(m["hard_sums"]):
        expected = 1 if player_sum < 15 else 0
        assert (m["hard"][i] == expected).all()


# --------------------------------------------------------------------------
# strategy.render_matrix
# --------------------------------------------------------------------------


def test_render_matrix_headers_and_rows():
    text = st.render_matrix(st.decision_matrix(stub_hit_below_15), "Agent strategy")
    assert text
    assert "Agent strategy" in text
    assert "Hard totals" in text and "Soft totals" in text
    assert "Legend" in text
    header_line = [l for l in text.splitlines() if l.strip().startswith("total")][0]
    assert header_line.split("|")[1].split() == ["A"] + [str(u) for u in range(2, 11)]
    assert any(l.strip().startswith("21 ") or l.strip().startswith("21|") for l in text.splitlines())


def test_render_matrix_marks_disagreements():
    agent = st.decision_matrix(stub_stand)
    reference = st.decision_matrix(stub_hit_below_15)
    text = st.render_matrix(agent, "Agent vs optimal", reference=reference)
    assert "*" in text
    assert "s*" in text  # lowercased stand cell where the reference hits
    # 11 hard rows (4..14) and 3 soft rows (12..14) differ, 10 columns each.
    assert "Disagreements: 140 / 280 cells" in text

    same = st.render_matrix(agent, "same", reference=agent)
    assert "Disagreements: 0 / 280 cells" in same
    assert "*" not in same.split("Legend")[0]


def test_render_matrix_accepts_a_bare_grid():
    m = st.decision_matrix(stub_stand)
    text = st.render_matrix(m["soft"], "Soft only")
    assert "Soft only" in text
    assert "12" in text and "21" in text


# --------------------------------------------------------------------------
# strategy.alignment -- against a hand-built OptimalSolution stand-in
# --------------------------------------------------------------------------


class _FakeOptimal:
    """Duck-type of optimal.OptimalSolution, small enough to reason about."""

    def __init__(self, policy, state_frequency, state_value=None, values=None):
        self.policy = policy
        self.state_frequency = state_frequency
        self.state_value = state_value or {}
        self.expected_value = 0.0
        self._values = values  # state -> {action: value}

    def action_values(self, state):
        if self._values is None:
            raise AttributeError("no action values")
        return self._values[state]


def _fake_optimal(with_values=True):
    # Four states: two stand-optimal, two hit-optimal, with lopsided frequencies.
    policy = {
        (20, 5, 0): 0,
        (19, 10, 0): 0,
        (12, 7, 0): 1,
        (14, 10, 0): 1,
    }
    freq = {
        (20, 5, 0): 0.50,
        (19, 10, 0): 0.30,
        (12, 7, 0): 0.15,
        (14, 10, 0): 0.05,
    }
    values = {
        (20, 5, 0): {0: 0.65, 1: -0.80},   # ev_cost of hitting = 1.45
        (19, 10, 0): {0: 0.28, 1: -0.75},  # ev_cost of hitting = 1.03
        (12, 7, 0): {0: -0.48, 1: -0.21},  # ev_cost of standing = 0.27
        (14, 10, 0): {0: -0.58, 1: -0.54}, # ev_cost of standing = 0.04
    }
    state_value = {s: v[policy[s]] for s, v in values.items()}
    return _FakeOptimal(policy, freq, state_value, values if with_values else None)


def _policy_from(table, default=0):
    def fn(state, rng):
        return table.get(tuple(state), default)

    return fn


def test_alignment_perfect_match():
    opt = _fake_optimal()
    result = st.alignment(_policy_from(opt.policy), opt)
    assert result["raw"] == pytest.approx(1.0)
    assert result["weighted"] == pytest.approx(1.0)
    assert result["disagreements"] == []
    assert result["n_states"] == 4


def test_alignment_exact_opposite():
    opt = _fake_optimal()
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    result = st.alignment(_policy_from(flipped), opt)
    assert result["raw"] == pytest.approx(0.0)
    assert result["weighted"] == pytest.approx(0.0)
    assert len(result["disagreements"]) == 4


def test_alignment_partial_is_frequency_weighted():
    opt = _fake_optimal()
    # Agree on the two stand states (freq 0.50 + 0.30), disagree on both hits.
    table = dict(opt.policy)
    table[(12, 7, 0)] = 0
    table[(14, 10, 0)] = 0
    result = st.alignment(_policy_from(table), opt)
    assert result["raw"] == pytest.approx(0.5)
    assert result["weighted"] == pytest.approx(0.80)

    # And the reverse split: agree only on the rare hit states.
    table2 = {s: 1 - a for s, a in opt.policy.items()}
    table2[(12, 7, 0)] = 1
    table2[(14, 10, 0)] = 1
    result2 = st.alignment(_policy_from(table2), opt)
    assert result2["raw"] == pytest.approx(0.5)
    assert result2["weighted"] == pytest.approx(0.20)


def test_alignment_disagreements_are_sorted_by_cost():
    opt = _fake_optimal()
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    result = st.alignment(_policy_from(flipped), opt)

    d = result["disagreements"]
    # Positional unpacking must follow the documented contract order.
    state, agent_action, optimal_action, freq, ev_cost = d[0]
    assert state == (20, 5, 0)
    assert agent_action == 1 and optimal_action == 0
    assert freq == pytest.approx(0.50)
    assert ev_cost == pytest.approx(1.45)

    keys = [x.frequency * x.ev_cost for x in d]
    assert keys == sorted(keys, reverse=True)
    assert [x.state for x in d] == [(20, 5, 0), (19, 10, 0), (12, 7, 0), (14, 10, 0)]


def test_alignment_without_action_values_falls_back_to_zero_cost():
    opt = _fake_optimal(with_values=False)
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    result = st.alignment(_policy_from(flipped), opt)
    assert len(result["disagreements"]) == 4
    assert all(d.ev_cost == 0.0 for d in result["disagreements"])
    # Scores are unaffected by the missing accessor.
    assert result["raw"] == pytest.approx(0.0)


def test_alignment_accepts_sequence_action_values():
    opt = _fake_optimal()
    opt._values = {s: [v[0], v[1]] for s, v in opt._values.items()}
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    result = st.alignment(_policy_from(flipped), opt)
    assert result["disagreements"][0].ev_cost == pytest.approx(1.45)


def test_alignment_handles_zero_total_frequency():
    opt = _fake_optimal()
    opt.state_frequency = {s: 0.0 for s in opt.policy}
    result = st.alignment(_policy_from(opt.policy), opt)
    assert result["raw"] == pytest.approx(1.0)
    assert result["weighted"] == 0.0


def test_alignment_scores_a_real_policy_fn():
    opt = _fake_optimal()
    # dealer_mimic hits below 17: agrees on the two hit states and the two
    # stand states (19, 20 are both >= 17), so it matches everywhere here.
    result = st.alignment(ev.dealer_mimic, opt)
    assert result["raw"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# strategy.render_alignment
# --------------------------------------------------------------------------


def test_render_alignment_contains_headers():
    opt = _fake_optimal()
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    text = st.render_alignment(st.alignment(_policy_from(flipped), opt))
    assert text
    assert "Raw agreement" in text
    assert "Weighted agreement" in text
    assert "EV cost" in text
    assert "0.00%" in text
    lines = text.splitlines()
    # Most costly mistake is listed first.
    body = [l for l in lines if l.strip().startswith("20 ")]
    assert body


def test_render_alignment_top_n_truncates():
    opt = _fake_optimal()
    flipped = {s: 1 - a for s, a in opt.policy.items()}
    result = st.alignment(_policy_from(flipped), opt)
    text = st.render_alignment(result, top_n=2)
    assert "Top 2 most costly" in text
    assert "and 2 more" in text


def test_render_alignment_perfect_policy():
    opt = _fake_optimal()
    text = st.render_alignment(st.alignment(_policy_from(opt.policy), opt))
    assert "100.00%" in text
    assert "No disagreements" in text
