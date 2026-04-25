import json
import os
import pytest
import tempfile

# Add project root to path so we can import model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model

REQUIRED_KEYS = [
    "H", "phi", "K", "n_simulations", "random_seed",
    "n_prefilter", "min_bootstrap_samples",
    "trend_short_H", "trend_long_H",
    "trend_z_threshold", "fade_z_threshold",
]

def make_params_file(data, tmp_path):
    """Write a dict to a temp params file and return the path."""
    p = tmp_path / "params.json"
    p.write_text(json.dumps(data))
    return str(p)

def valid_params():
    return {
        "H": 10, "phi": 0.7, "K": 10,
        "n_simulations": 10000, "random_seed": 42,
        "n_prefilter": 20, "min_bootstrap_samples": 3,
        "trend_short_H": 4, "trend_long_H": 12,
        "trend_z_threshold": 1.0, "fade_z_threshold": 2.0,
    }

def test_load_config_valid(tmp_path):
    path = make_params_file(valid_params(), tmp_path)
    cfg = model.load_config(path)
    assert cfg["H"] == 10
    assert cfg["phi"] == 0.7

def test_load_config_missing_key(tmp_path):
    data = valid_params()
    del data["H"]
    path = make_params_file(data, tmp_path)
    with pytest.raises(ValueError, match="Missing required params"):
        model.load_config(path)

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        model.load_config("nonexistent_params.json")


def test_decay_weight_most_recent():
    # Race from 0 races ago, same season: weight should be 1.0
    w = model.decay_weight(delta_r=0, N=0, H=10, phi=0.7)
    assert w == pytest.approx(1.0)

def test_decay_weight_half_life():
    # Race from exactly H races ago, same season: weight should be 0.5
    w = model.decay_weight(delta_r=10, N=0, H=10, phi=0.7)
    assert w == pytest.approx(0.5)

def test_decay_weight_season_boundary():
    # One season boundary crossed applies phi multiplier
    w_no_boundary = model.decay_weight(delta_r=10, N=0, H=10, phi=0.7)
    w_boundary    = model.decay_weight(delta_r=10, N=1, H=10, phi=0.7)
    assert w_boundary == pytest.approx(w_no_boundary * 0.7)

def test_decay_weight_two_seasons():
    # Two boundaries: phi^2
    w = model.decay_weight(delta_r=0, N=2, H=10, phi=0.7)
    assert w == pytest.approx(0.7 ** 2)
