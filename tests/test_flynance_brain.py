"""Tests for the fly-brain network (flynance/brain.py) and its sensory encoding.

Covers success criteria S1 (exactly 4,010 parameters) and S2 (analytic
gradients match finite differences to < 1e-6).

Run from the repository root::

    python -m pytest tests/test_flynance_brain.py -v
"""

import os
import warnings

import numpy as np
import pytest

# Add project root to path so we can import flynance
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance.brain import PARAM_NAMES, FlyBlackjackBrain
from flynance.encoding import (
    INPUT_SIZE,
    RICH_INPUT_SIZE,
    encode_batch,
    preprocess_state,
    preprocess_state_rich,
)

FD_STEP = 1e-5          # finite-difference h
FD_TOLERANCE = 1e-6     # max relative error allowed between analytic and numeric


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def make_brain(seed=0, **kwargs):
    return FlyBlackjackBrain(np.random.default_rng(seed), **kwargs)


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits, targets):
    """Mean cross-entropy of `logits` (B,2) against integer `targets` (B,)."""
    probs = softmax_rows(logits)
    picked = probs[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(picked)))


def cross_entropy_dlogits(logits, targets):
    """dLoss/dlogits for the mean cross-entropy above; shape (B,2)."""
    probs = softmax_rows(logits)
    probs[np.arange(len(targets)), targets] -= 1.0
    return probs / len(targets)


def relative_error(a, b):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


# ----------------------------------------------------------------------
# architecture and parameter count (S1)
# ----------------------------------------------------------------------
def test_layer_sizes_are_the_spec_architecture():
    assert FlyBlackjackBrain.LAYER_SIZES == (3, 64, 56, 2)
    assert FlyBlackjackBrain.N_PARAMETERS == 4010


def test_num_parameters_is_exactly_4010():
    brain = make_brain()
    assert brain.num_parameters() == 4010
    assert brain.num_parameters() == FlyBlackjackBrain.N_PARAMETERS


def test_individual_tensor_shapes():
    brain = make_brain()
    expected = {
        "W1": (3, 64),
        "b1": (64,),
        "W2": (64, 56),
        "b2": (56,),
        "W3": (56, 2),
        "b3": (2,),
    }
    assert set(brain.params) == set(expected)
    for name, shape in expected.items():
        assert brain.params[name].shape == shape, name
        assert brain.params[name].dtype == np.float64, name
        assert getattr(brain, name).shape == shape, name
    # The table in the spec, summed.
    assert sum(np.prod(s) for s in expected.values()) == 4010


def test_biases_start_at_zero_and_weights_do_not():
    brain = make_brain()
    for name in ("b1", "b2", "b3"):
        assert np.all(brain.params[name] == 0.0)
    for name in ("W1", "W2", "W3"):
        assert np.any(brain.params[name] != 0.0)


def test_he_normal_scale_is_sqrt_2_over_fan_in():
    # Large-sample check on the widest tensor: std should be ~sqrt(2/64).
    brain = make_brain(seed=123)
    expected_std = np.sqrt(2.0 / 64)
    observed = brain.params["W2"].std()
    assert abs(observed - expected_std) < 0.1 * expected_std


def test_biological_aliases_point_at_the_same_arrays():
    brain = make_brain()
    assert brain.sensory_layer_w is brain.params["W1"]
    assert brain.sensory_layer_b is brain.params["b1"]
    assert brain.mushroom_body_w is brain.params["W2"]
    assert brain.mushroom_body_b is brain.params["b2"]
    assert brain.motor_layer_w is brain.params["W3"]
    assert brain.motor_layer_b is brain.params["b3"]


# ----------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------
def test_forward_shape_and_batch_of_one():
    brain = make_brain()
    x = encode_batch([(20, 1, 1), (12, 10, 0), (16, 6, 0)])
    assert brain.forward(x).shape == (3, 2)
    single = preprocess_state((18, 4, 0)).reshape(1, 3)
    assert brain.forward(single).shape == (1, 2)


def test_forward_rejects_wrong_input_width():
    brain = make_brain()
    with pytest.raises(ValueError):
        brain.forward(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        brain.forward(np.zeros(3))


def test_forward_matches_a_manual_recomputation():
    brain = make_brain(seed=7)
    x = encode_batch([(15, 7, 0), (21, 1, 1)])
    a1 = np.maximum(x @ brain.W1 + brain.b1, 0.0)
    a2 = np.maximum(a1 @ brain.W2 + brain.b2, 0.0)
    expected = a2 @ brain.W3 + brain.b3
    assert np.allclose(brain.forward(x), expected, rtol=0, atol=0)


# ----------------------------------------------------------------------
# gradient check (S2)
# ----------------------------------------------------------------------
def _fd_indices(name, param, rng):
    """Every entry of the small tensors; a subsample of the 3,584-entry W2."""
    flat = param.size
    if name == "W2":
        return rng.choice(flat, size=64, replace=False)
    return np.arange(flat)


def test_analytic_gradients_match_finite_differences():
    rng = np.random.default_rng(2024)
    brain = make_brain(seed=11)

    states = [(20, 1, 1), (12, 10, 0), (16, 6, 0), (17, 7, 1), (4, 2, 0), (21, 5, 1)]
    x = encode_batch(states)
    targets = np.array([0, 1, 1, 0, 1, 0])

    # Analytic gradients.
    logits = brain.forward(x)
    brain.backward(cross_entropy_dlogits(logits, targets))
    analytic = {name: brain.grads[name].copy() for name in PARAM_NAMES}

    def loss_now():
        return cross_entropy(brain.forward(x), targets)

    worst = 0.0
    worst_where = None
    checked = {name: 0 for name in PARAM_NAMES}

    for name in PARAM_NAMES:
        param = brain.params[name]
        flat = param.reshape(-1)
        grad_flat = analytic[name].reshape(-1)
        for idx in _fd_indices(name, param, rng):
            original = flat[idx]

            flat[idx] = original + FD_STEP
            loss_plus = loss_now()
            flat[idx] = original - FD_STEP
            loss_minus = loss_now()
            flat[idx] = original

            numeric = (loss_plus - loss_minus) / (2.0 * FD_STEP)
            err = relative_error(numeric, grad_flat[idx])
            checked[name] += 1
            if err > worst:
                worst, worst_where = err, (name, int(idx))

    # All six tensors were exercised.
    for name in PARAM_NAMES:
        assert checked[name] > 0, name
    assert checked["W1"] == 192 and checked["b1"] == 64
    assert checked["b2"] == 56 and checked["W3"] == 112 and checked["b3"] == 2

    assert worst < FD_TOLERANCE, f"max relative error {worst:.3e} at {worst_where}"


# ----------------------------------------------------------------------
# gradient accumulation
# ----------------------------------------------------------------------
def test_backward_accumulates_across_calls():
    brain = make_brain(seed=3)
    x = encode_batch([(18, 9, 0), (13, 3, 0)])
    dlogits = np.array([[0.4, -0.4], [-0.2, 0.2]])

    brain.forward(x)
    brain.backward(dlogits)
    once = {name: brain.grads[name].copy() for name in PARAM_NAMES}

    brain.backward(dlogits)  # same cache, no step in between
    for name in PARAM_NAMES:
        assert np.allclose(brain.grads[name], 2.0 * once[name])
        assert np.any(once[name] != 0.0), name


def test_zero_grad_clears_accumulators():
    brain = make_brain(seed=3)
    x = encode_batch([(18, 9, 0)])
    brain.forward(x)
    brain.backward(np.array([[1.0, -1.0]]))
    assert any(np.any(brain.grads[n] != 0.0) for n in PARAM_NAMES)
    brain.zero_grad()
    for name in PARAM_NAMES:
        assert np.all(brain.grads[name] == 0.0)


def test_backward_before_forward_raises():
    brain = make_brain()
    with pytest.raises(RuntimeError):
        brain.backward(np.zeros((1, 2)))


def test_backward_rejects_mismatched_shape():
    brain = make_brain()
    brain.forward(encode_batch([(18, 9, 0)]))
    with pytest.raises(ValueError):
        brain.backward(np.zeros((2, 2)))


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------
def test_step_changes_parameters_and_zeroes_gradients():
    brain = make_brain(seed=5)
    before = {name: brain.params[name].copy() for name in PARAM_NAMES}

    x = encode_batch([(19, 8, 0), (14, 4, 0)])
    logits = brain.forward(x)
    brain.backward(cross_entropy_dlogits(logits, np.array([1, 0])))
    brain.step()

    assert brain.t == 1
    for name in PARAM_NAMES:
        assert np.all(brain.grads[name] == 0.0), name
    assert any(
        np.any(brain.params[name] != before[name]) for name in PARAM_NAMES
    )


def test_first_adam_step_moves_about_lr_for_nonzero_gradients():
    lr = 0.005
    brain = make_brain(seed=5, lr=lr)
    before = {name: brain.params[name].copy() for name in PARAM_NAMES}

    x = encode_batch([(19, 8, 0), (14, 4, 0), (11, 2, 0)])
    logits = brain.forward(x)
    brain.backward(cross_entropy_dlogits(logits, np.array([1, 0, 1])))
    grads = {name: brain.grads[name].copy() for name in PARAM_NAMES}
    brain.step()

    moved_any = False
    for name in PARAM_NAMES:
        delta = np.abs(brain.params[name] - before[name])
        nonzero = np.abs(grads[name]) > 1e-9
        if not np.any(nonzero):
            continue
        moved_any = True
        # First Adam step: m_hat / sqrt(v_hat) == sign(g), so |delta| == lr,
        # up to the eps damping of lr * |g| / (|g| + eps).
        assert np.allclose(delta[nonzero], lr, rtol=1e-3, atol=0.0), name
        # Parameters with no gradient must not move at all.
        assert np.all(delta[~nonzero] == 0.0), name
    assert moved_any


def test_timestep_advances_once_per_step_not_per_backward():
    brain = make_brain(seed=5)
    x = encode_batch([(19, 8, 0)])
    for _ in range(3):
        brain.forward(x)
        brain.backward(np.array([[0.5, -0.5]]))
    brain.step()
    assert brain.t == 1
    brain.forward(x)
    brain.backward(np.array([[0.5, -0.5]]))
    brain.step()
    assert brain.t == 2


def test_repeated_steps_reduce_a_supervised_loss():
    brain = make_brain(seed=9)
    x = encode_batch([(20, 1, 1), (12, 10, 0), (16, 6, 0), (17, 7, 1)])
    targets = np.array([0, 1, 1, 0])

    first = cross_entropy(brain.forward(x), targets)
    for _ in range(100):
        logits = brain.forward(x)
        brain.backward(cross_entropy_dlogits(logits, targets))
        brain.step()
    last = cross_entropy(brain.forward(x), targets)
    assert last < first


# ----------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------
def test_policy_rows_are_positive_and_sum_to_one():
    brain = make_brain(seed=13)
    x = encode_batch([(20, 1, 1), (12, 10, 0), (4, 2, 0)])
    probs = brain.policy(x)
    assert probs.shape == (3, 2)
    assert np.all(probs > 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_policy_is_stable_for_huge_logits():
    brain = make_brain(seed=13)
    # Force enormous logits by scaling up the motor layer.
    brain.params["W3"] *= 1e3
    brain.params["b3"] += 1e3
    x = encode_batch([(20, 1, 1), (12, 10, 0)])
    assert np.max(np.abs(brain.forward(x))) > 1e3

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        with np.errstate(over="raise", invalid="raise"):
            probs = brain.policy(x)

    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------
def test_same_seed_gives_identical_parameters():
    a = make_brain(seed=42)
    b = make_brain(seed=42)
    for name in PARAM_NAMES:
        assert np.array_equal(a.params[name], b.params[name]), name


def test_different_seeds_give_different_parameters():
    a = make_brain(seed=42)
    b = make_brain(seed=43)
    assert not np.array_equal(a.params["W1"], b.params["W1"])
    assert not np.array_equal(a.params["W2"], b.params["W2"])
    assert not np.array_equal(a.params["W3"], b.params["W3"])


def test_init_requires_a_generator():
    with pytest.raises(TypeError):
        FlyBlackjackBrain(42)


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------
def test_save_load_round_trip(tmp_path):
    brain = make_brain(seed=17, lr=0.003, beta1=0.85, beta2=0.99, eps=1e-7)
    x = encode_batch([(20, 1, 1), (12, 10, 0), (16, 6, 0)])

    # Take a couple of steps so Adam state is non-trivial.
    for _ in range(3):
        logits = brain.forward(x)
        brain.backward(cross_entropy_dlogits(logits, np.array([0, 1, 1])))
        brain.step()

    path = tmp_path / "brain.npz"
    brain.save(path)
    reloaded = FlyBlackjackBrain.load(path)

    for name in PARAM_NAMES:
        assert np.array_equal(reloaded.params[name], brain.params[name]), name
        assert np.array_equal(reloaded.m[name], brain.m[name]), name
        assert np.array_equal(reloaded.v[name], brain.v[name]), name
    assert reloaded.t == brain.t
    assert (reloaded.lr, reloaded.beta1, reloaded.beta2, reloaded.eps) == (
        brain.lr,
        brain.beta1,
        brain.beta2,
        brain.eps,
    )
    assert reloaded.num_parameters() == 4010
    assert np.array_equal(reloaded.forward(x), brain.forward(x))


def test_reloaded_brain_continues_training_identically(tmp_path):
    brain = make_brain(seed=17)
    x = encode_batch([(20, 1, 1), (12, 10, 0)])
    targets = np.array([0, 1])

    logits = brain.forward(x)
    brain.backward(cross_entropy_dlogits(logits, targets))
    brain.step()

    path = tmp_path / "brain.npz"
    brain.save(path)
    reloaded = FlyBlackjackBrain.load(path)

    for b in (brain, reloaded):
        logits = b.forward(x)
        b.backward(cross_entropy_dlogits(logits, targets))
        b.step()

    for name in PARAM_NAMES:
        assert np.array_equal(reloaded.params[name], brain.params[name]), name


def test_load_does_not_alias_the_saved_arrays(tmp_path):
    brain = make_brain(seed=17)
    path = tmp_path / "brain.npz"
    brain.save(path)
    reloaded = FlyBlackjackBrain.load(path)
    reloaded.params["W1"] += 1.0
    assert not np.array_equal(reloaded.params["W1"], brain.params["W1"])
    assert np.all(reloaded.grads["W1"] == 0.0)


# ----------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------
def test_preprocess_state_is_the_spec_mapping():
    v = preprocess_state((20, 1, 1))
    assert v.shape == (3,)
    assert v.dtype == np.float64
    assert np.array_equal(v, np.array([20 / 21.0, 0.1, 1.0]))


@pytest.mark.parametrize(
    "state,expected",
    [
        ((20, 1, 1), [20 / 21.0, 1 / 10.0, 1.0]),
        ((12, 10, 0), [12 / 21.0, 1.0, 0.0]),
        ((4, 2, 0), [4 / 21.0, 0.2, 0.0]),
        ((21, 5, 1), [1.0, 0.5, 1.0]),
    ],
)
def test_preprocess_state_known_values(state, expected):
    assert np.allclose(preprocess_state(state), expected, rtol=0, atol=0)


def test_encode_batch_shape_and_agreement_with_single_calls():
    states = [(20, 1, 1), (12, 10, 0), (16, 6, 0), (4, 2, 0)]
    batch = encode_batch(states)
    assert batch.shape == (len(states), INPUT_SIZE)
    assert batch.dtype == np.float64
    for i, s in enumerate(states):
        assert np.array_equal(batch[i], preprocess_state(s))


def test_encode_batch_handles_empty_input():
    assert encode_batch([]).shape == (0, 3)


def test_encoded_batch_feeds_the_brain_directly():
    brain = make_brain()
    states = [(20, 1, 1), (12, 10, 0)]
    assert brain.forward(encode_batch(states)).shape == (2, 2)


def test_rich_encoding_one_hots_the_dealer_upcard():
    assert RICH_INPUT_SIZE == 12
    ace = preprocess_state_rich((20, 1, 1))
    assert ace.shape == (RICH_INPUT_SIZE,)
    assert ace[0] == 20 / 21.0
    assert ace[1] == 1.0            # dealer ace
    assert ace[11] == 1.0           # usable ace
    assert ace[2:11].sum() == 0.0

    ten = preprocess_state_rich((12, 10, 0))
    assert ten[10] == 1.0
    assert ten[1] == 0.0
    assert ten[11] == 0.0
    assert ten[1:11].sum() == 1.0


def test_rich_encoding_separates_ace_from_two():
    """The spec encoding puts a dealer ace (0.1) right next to a two (0.2)."""
    spec_ace = preprocess_state((16, 1, 0))
    spec_two = preprocess_state((16, 2, 0))
    assert abs(spec_ace[1] - spec_two[1]) == pytest.approx(0.1)

    rich_ace = preprocess_state_rich((16, 1, 0))
    rich_two = preprocess_state_rich((16, 2, 0))
    # Orthogonal one-hots: equidistant from every other upcard.
    assert float(rich_ace @ rich_two) == pytest.approx(rich_ace[0] * rich_two[0])
