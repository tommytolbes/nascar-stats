"""Tests for the browser export.

The casino page reimplements the network in JavaScript, so the export it reads
has to stay faithful to the trained weights. The exact JS-vs-Python comparison
lives in flynance/casino/verify_browser_brain.mjs (it needs node); these tests
cover the Python half -- that the export is complete, well formed, and rounds
without changing any decision.
"""

import json
import os
import re
import sys

import numpy as np
import pytest

# Add project root to path so we can import flynance (same bootstrap as test_model.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flynance.brain import FlyBlackjackBrain
from flynance.encoding import preprocess_state
from flynance.export_web import PARAM_KEYS, export, write_reference


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """An untrained brain is enough: the export never inspects weight quality."""
    path = tmp_path_factory.mktemp("model") / "brain.npz"
    FlyBlackjackBrain(rng=np.random.default_rng(0)).save(path)
    return path


def _parse(js_text, variable):
    """Pull one `window.X = {...};` assignment back out of the generated file."""
    match = re.search(rf"window\.{variable} = (.*?);\n", js_text, re.S)
    assert match, f"{variable} missing from the export"
    return json.loads(match.group(1))


def test_export_contains_all_4010_weights(model, tmp_path):
    out = export(model, tmp_path / "brain-data.js")
    weights = _parse(out.read_text(), "FLY_WEIGHTS")

    assert set(weights) == set(PARAM_KEYS)
    assert sum(len(v) for v in weights.values()) == FlyBlackjackBrain.N_PARAMETERS
    assert len(weights["W1"]) == 3 * 64
    assert len(weights["W2"]) == 64 * 56
    assert len(weights["W3"]) == 56 * 2


def test_export_is_row_major_matching_the_js_indexing(model, tmp_path):
    """The JS reads w[i * outN + j]; the export must be flattened to match."""
    out = export(model, tmp_path / "brain-data.js")
    weights = _parse(out.read_text(), "FLY_WEIGHTS")
    original = np.load(model)["param_W1"]

    flat = np.asarray(weights["W1"]).reshape(3, 64)
    assert np.allclose(flat, original, atol=1e-6)


def test_export_includes_the_exact_optimum(model, tmp_path):
    out = export(model, tmp_path / "brain-data.js")
    text = out.read_text()
    policy = _parse(text, "OPTIMAL_POLICY")

    assert len(policy) == 280
    assert set(policy.values()) <= {0, 1}
    assert policy["20_10_0"] == 0   # stand on a hard 20
    assert policy["12_10_0"] == 1   # hit 12 against a ten
    assert "window.OPTIMAL_EV = -0.0465" in text


def test_rounding_never_changes_a_decision(model, tmp_path):
    """6-decimal rounding keeps the file small; it must not alter any action."""
    out = export(model, tmp_path / "brain-data.js")
    weights = {k: np.asarray(v) for k, v in _parse(out.read_text(), "FLY_WEIGHTS").items()}

    rounded = FlyBlackjackBrain(rng=np.random.default_rng(0))
    for key in PARAM_KEYS:
        rounded.params[key] = weights[key].reshape(rounded.params[key].shape)

    exact = FlyBlackjackBrain.load(model)
    for player_sum in range(4, 22):
        for upcard in range(1, 11):
            for ace in (0, 1):
                if ace and player_sum < 12:
                    continue
                x = preprocess_state((player_sum, upcard, ace))[None, :]
                assert exact.policy(x).argmax() == rounded.policy(x).argmax(), (
                    f"rounding flipped the action at {(player_sum, upcard, ace)}"
                )


def test_reference_covers_every_decision_state(model, tmp_path):
    path = write_reference(model, tmp_path / "reference_policy.json")
    reference = json.loads(path.read_text())

    assert len(reference) == 280
    for probs in reference.values():
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-5
