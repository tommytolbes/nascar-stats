"""The fly brain: a 4,010-parameter 3 -> 64 -> 56 -> 2 ReLU network in pure NumPy.

Layers follow the biological framing of the design spec:

===============  ==========  ========================================
Layer            Shape       Analogue
===============  ==========  ========================================
sensory layer    3 -> 64     antennal lobe / Kenyon cell projection
mushroom body    64 -> 56    mushroom body intrinsic sub-network
motor layer      56 -> 2     descending motor neurons (0=STAND, 1=HIT)
===============  ==========  ========================================

Forward pass, backward pass and the Adam optimizer are all hand-written; there
is no autodiff framework and no PyTorch dependency (see the spec's "Why NumPy
and not PyTorch").  Everything is ``float64``, which keeps the finite-difference
gradient check in ``tests/test_flynance_brain.py`` meaningful.

Usage::

    rng = np.random.default_rng(0)
    brain = FlyBlackjackBrain(rng)
    logits = brain.forward(x)          # (B, 3) -> (B, 2)
    brain.backward(dlogits)            # accumulates gradients
    brain.backward(more_dlogits)       # accumulates again
    brain.step()                       # one Adam update, then zero_grad()
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["FlyBlackjackBrain", "PARAM_NAMES"]

#: Canonical parameter order, matching the spec's parameter-count table.
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


class FlyBlackjackBrain:
    """A three-layer ReLU policy network with a hand-written Adam optimizer.

    Parameters are stored in :attr:`params`, a dict keyed by
    ``"W1", "b1", "W2", "b2", "W3", "b3"``; the same arrays are reachable as
    attributes (``brain.W1``) and under their biological names
    (``brain.sensory_layer_w`` ...).

    Attributes
    ----------
    params : dict[str, np.ndarray]
        The 4,010 learnable parameters.
    grads : dict[str, np.ndarray]
        Gradient accumulators, same shapes; :meth:`backward` adds into these
        and :meth:`step` / :meth:`zero_grad` clear them.
    t : int
        Adam timestep, incremented once per :meth:`step`.
    """

    #: Layer widths: input, sensory, mushroom body, motor.
    LAYER_SIZES = (3, 64, 56, 2)

    #: Authoritative parameter count (the spec's own 4,180 / "~4,184" is wrong).
    N_PARAMETERS = 4010

    #: Action indices, matching gymnasium's Blackjack-v1.
    STAND = 0
    HIT = 1

    def __init__(
        self,
        rng: np.random.Generator,
        lr: float = 0.005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Build a brain with He/Kaiming-normal weights and zero biases.

        Parameters
        ----------
        rng:
            Injected generator.  Every random draw comes from it, so two brains
            built from equally-seeded generators are bit-identical.
        lr, beta1, beta2, eps:
            Adam hyperparameters.  Defaults are the specification's.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        self.rng = rng
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        n_in, n_sensory, n_mushroom, n_motor = self.LAYER_SIZES

        self.params: dict[str, np.ndarray] = {
            "W1": self._he_normal(n_in, n_sensory),
            "b1": np.zeros(n_sensory, dtype=np.float64),
            "W2": self._he_normal(n_sensory, n_mushroom),
            "b2": np.zeros(n_mushroom, dtype=np.float64),
            "W3": self._he_normal(n_mushroom, n_motor),
            "b3": np.zeros(n_motor, dtype=np.float64),
        }

        # Gradient accumulators and Adam moment estimates.
        self.grads: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.m: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.v: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.t: int = 0

        self._cache: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _he_normal(self, fan_in: int, fan_out: int) -> np.ndarray:
        """He/Kaiming normal weights: ``N(0, sqrt(2 / fan_in))``, shape ``(fan_in, fan_out)``."""
        std = np.sqrt(2.0 / fan_in)
        return (self.rng.standard_normal((fan_in, fan_out)) * std).astype(
            np.float64, copy=False
        )

    # ------------------------------------------------------------------
    # named views onto the parameters
    # ------------------------------------------------------------------
    @property
    def W1(self) -> np.ndarray:  # noqa: N802 - matches the spec's table
        """Sensory-layer weights, ``(3, 64)``."""
        return self.params["W1"]

    @property
    def b1(self) -> np.ndarray:
        """Sensory-layer biases, ``(64,)``."""
        return self.params["b1"]

    @property
    def W2(self) -> np.ndarray:  # noqa: N802
        """Mushroom-body weights, ``(64, 56)``."""
        return self.params["W2"]

    @property
    def b2(self) -> np.ndarray:
        """Mushroom-body biases, ``(56,)``."""
        return self.params["b2"]

    @property
    def W3(self) -> np.ndarray:  # noqa: N802
        """Motor-layer weights, ``(56, 2)``."""
        return self.params["W3"]

    @property
    def b3(self) -> np.ndarray:
        """Motor-layer biases, ``(2,)``."""
        return self.params["b3"]

    # Biological aliases, for code that reads better with them.
    sensory_layer_w = W1
    sensory_layer_b = b1
    mushroom_body_w = W2
    mushroom_body_b = b2
    motor_layer_w = W3
    motor_layer_b = b3

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the network and cache what :meth:`backward` needs.

        Parameters
        ----------
        x:
            Shape ``(B, 3)``.  A single state is passed as ``(1, 3)``.

        Returns
        -------
        np.ndarray
            Raw logits, shape ``(B, 2)``.  No softmax -- see :meth:`policy`.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.LAYER_SIZES[0]:
            raise ValueError(
                f"forward expects shape (B, {self.LAYER_SIZES[0]}), got {x.shape}"
            )

        z1 = x @ self.params["W1"] + self.params["b1"]          # sensory pre-activation
        a1 = np.maximum(z1, 0.0)                                # sensory ReLU
        z2 = a1 @ self.params["W2"] + self.params["b2"]         # mushroom body pre-activation
        a2 = np.maximum(z2, 0.0)                                # mushroom body ReLU
        logits = a2 @ self.params["W3"] + self.params["b3"]     # motor layer

        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        """Backpropagate an upstream gradient and *accumulate* into :attr:`grads`.

        Gradients are added, never replaced, so several backward passes can be
        summed before a single :meth:`step` (e.g. accumulating a whole episode
        or a batch of episodes).  Call :meth:`zero_grad` to start fresh.

        Parameters
        ----------
        dlogits:
            ``dLoss/dlogits``, shape ``(B, 2)``, matching the most recent
            :meth:`forward`.  Any batch scaling (e.g. ``1/B``) is the caller's
            business; nothing is divided here.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        dlogits = np.asarray(dlogits, dtype=np.float64)
        cache = self._cache
        if dlogits.shape != (cache["x"].shape[0], self.LAYER_SIZES[-1]):
            raise ValueError(
                f"dlogits must have shape {(cache['x'].shape[0], self.LAYER_SIZES[-1])}, "
                f"got {dlogits.shape}"
            )

        x, z1, a1, z2, a2 = cache["x"], cache["z1"], cache["a1"], cache["z2"], cache["a2"]

        # Motor layer.
        self.grads["W3"] += a2.T @ dlogits
        self.grads["b3"] += dlogits.sum(axis=0)

        # Mushroom body.
        da2 = dlogits @ self.params["W3"].T
        dz2 = da2 * (z2 > 0.0)
        self.grads["W2"] += a1.T @ dz2
        self.grads["b2"] += dz2.sum(axis=0)

        # Sensory layer.
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0.0)
        self.grads["W1"] += x.T @ dz1
        self.grads["b1"] += dz1.sum(axis=0)

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------
    def zero_grad(self) -> None:
        """Reset every gradient accumulator to zero."""
        for g in self.grads.values():
            g.fill(0.0)

    def step(self) -> None:
        """Apply one bias-corrected Adam update, then zero the gradients.

        The timestep :attr:`t` advances once per call, not once per
        :meth:`backward`, so accumulating several backward passes into one step
        behaves like a single larger batch.
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for name in PARAM_NAMES:
            g = self.grads[name]
            m = self.m[name]
            v = self.v[name]

            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)

            m_hat = m / bc1
            v_hat = v / bc2
            self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        self.zero_grad()

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------
    def policy(self, x: np.ndarray) -> np.ndarray:
        """Action probabilities, shape ``(B, 2)``.

        Numerically stable softmax: the row maximum is subtracted before
        exponentiating, so logits of magnitude 1e3 or larger produce neither
        overflow warnings nor NaNs.
        """
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """Count the learnable parameters from the arrays themselves."""
        return int(sum(p.size for p in self.params.values()))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Write parameters, Adam state and hyperparameters to an ``.npz`` file.

        Everything needed for an identical continuation is stored: the six
        parameter tensors, both Adam moment buffers, the timestep and the
        hyperparameters.
        """
        payload: dict[str, np.ndarray] = {}
        for name in PARAM_NAMES:
            payload[f"param_{name}"] = self.params[name]
            payload[f"m_{name}"] = self.m[name]
            payload[f"v_{name}"] = self.v[name]
        payload["t"] = np.array(self.t, dtype=np.int64)
        payload["hyper"] = np.array(
            [self.lr, self.beta1, self.beta2, self.eps], dtype=np.float64
        )
        np.savez(str(path), **payload)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FlyBlackjackBrain":
        """Rebuild a brain saved by :meth:`save`.

        The returned brain has identical parameters, Adam state and
        hyperparameters, so its forward output and its next update match the
        saved one exactly.  A fresh generator is attached (initialization is
        overwritten anyway, and the generator is not part of the checkpoint).
        """
        with np.load(str(path)) as data:
            hyper = np.asarray(data["hyper"], dtype=np.float64)
            brain = cls(
                np.random.default_rng(0),
                lr=float(hyper[0]),
                beta1=float(hyper[1]),
                beta2=float(hyper[2]),
                eps=float(hyper[3]),
            )
            for name in PARAM_NAMES:
                brain.params[name] = np.array(data[f"param_{name}"], dtype=np.float64)
                brain.m[name] = np.array(data[f"m_{name}"], dtype=np.float64)
                brain.v[name] = np.array(data[f"v_{name}"], dtype=np.float64)
            brain.grads = {k: np.zeros_like(v) for k, v in brain.params.items()}
            brain.t = int(data["t"])
        return brain

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sizes = " -> ".join(str(s) for s in self.LAYER_SIZES)
        return (
            f"FlyBlackjackBrain({sizes}, params={self.num_parameters()}, "
            f"lr={self.lr}, t={self.t})"
        )
