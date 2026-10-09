"""Two ways to train the fly brain, so the difference can be measured.

``train_spec_baseline`` reproduces the original project specification's training
loop exactly: one ``env.step`` per episode, ``loss = -log_prob(action) * reward``,
then reset. ``train_reinforce`` is the corrected version: full episodes,
return-to-go, a variance-reducing baseline and an entropy bonus.

Why the spec loop cannot work
-----------------------------
Blackjack-v1 returns ``reward = 0.0`` for a HIT that does not bust. The spec's
loss is then ``-log_prob * 0.0 = 0.0``, so no gradient flows at all. HIT
therefore receives signal *only* when it busts (-1), while STAND receives an
immediate +/-1 on every episode. The policy has no way to discover that hitting
a 12 is usually good, because the reward for the hand that hitting makes
possible is never attributed to the hit. States reached after a hit are never
visited either, since the loop resets immediately.

Measured consequence: the policy collapses onto a single constant action, and
*which* action is decided by the seed. Both actions look bad to this loop --
standing pays about -0.18 per hand, and hitting pays 0 (invisible) or -1 (bust)
-- so it is a race between two negative signals with no baseline to calibrate
them. Over seeds 0-7 at 20,000 episodes, four collapsed to always-hit, three to
always-stand, one landed in between. Always-hit is the catastrophic case: hitting
until the policy would stop never stops, so every hand busts and the EV is
exactly -1.0. ``stand_rate_on_hittable`` measures the collapse in either
direction.

Both trainers share one ``FlyBlackjackBrain`` implementation and one sensory
encoding, so the only difference between them is the credit-assignment rule.

Neuromodulatory framing (from the spec): a terminal reward of +1 is the
dopaminergic / sugar pulse, -1 is the octopaminergic / heat pulse, and 0 is
neutral. In the corrected trainer that pulse is propagated back over every
decision in the hand rather than only the last one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from flynance.brain import FlyBlackjackBrain
from flynance.encoding import preprocess_state

# Player totals at which standing is indefensible: a hand of 11 or less cannot
# bust, and 12-16 is where the whole hit/stand decision actually lives. Optimal
# play hits every one of these, so the stand-rate over them measures collapse.
HITTABLE_SUMS = tuple(range(4, 17))

# Maximum decisions in one hand; hitting always terminates eventually (bust), so
# this only guards against a bug, never against normal play.
MAX_STEPS_PER_HAND = 50

# Defaults for the corrected trainer, chosen by a sweep over learning rate,
# decay and entropy and scored on mean AND worst-case results across three
# seeds, so the choice is not one lucky run. See flynance/README.md "Tuning".
#
# The spec's lr=0.005 is far too hot here. Adam normalizes the gradient, so each
# step moves every weight by roughly lr no matter how noisy the estimate is; at
# 0.005 with one hand per step the policy never settles. Decaying to 2e-5 lets
# it explore early and converge late. The spec's own value is preserved where it
# belongs -- in train_spec_baseline, which must stay faithful.
TUNED_HYPERPARAMETERS = {"lr": 0.0005, "lr_final": 0.00002, "entropy_coef": 0.01}


@dataclass
class TrainingHistory:
    """Per-run record: rewards, periodic diagnostics, and the exact config used."""

    episode_rewards: list[float] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def moving_average(self, window: int = 1000) -> list[float]:
        """Mean episode reward over consecutive non-overlapping blocks."""
        rewards = np.asarray(self.episode_rewards, dtype=np.float64)
        n_blocks = len(rewards) // window
        if n_blocks == 0:
            return [float(rewards.mean())] if len(rewards) else []
        trimmed = rewards[: n_blocks * window].reshape(n_blocks, window)
        return [float(v) for v in trimmed.mean(axis=1)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _policy_gradient_dlogits(
    probs: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    entropy_coef: float,
) -> np.ndarray:
    """Gradient of the REINFORCE loss with respect to the motor-layer logits.

    For loss ``-w * log p_a`` the gradient is ``w * (p - onehot(a))``.

    The entropy bonus subtracts ``entropy_coef * H`` from the loss, and for a
    softmax ``dH/dz_j = -p_j * (log p_j + H)``, so it contributes
    ``+entropy_coef * p_j * (log p_j + H)`` to the gradient. Keeping entropy up
    stops the policy from collapsing onto one action before it has explored.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actions)), actions] = 1.0
    dlogits = weights[:, None] * (probs - onehot)

    if entropy_coef:
        log_probs = np.log(np.clip(probs, 1e-12, None))
        entropy = -(probs * log_probs).sum(axis=1, keepdims=True)
        dlogits += entropy_coef * probs * (log_probs + entropy)

    return dlogits


def _make_env() -> gym.Env:
    """The environment exactly as the specification configures it."""
    return gym.make("Blackjack-v1", natural=False, sab=False)


def stand_rate_on_hittable(brain: FlyBlackjackBrain) -> float:
    """Fraction of obviously-hittable states where the brain greedily stands.

    Swept over player totals 4-16, dealer upcards 1-10 and both ace flags.
    Optimal play hits every one of them, so a competent policy scores near 0.0.
    A value near 1.0 means the policy has collapsed to always-stand; near 0.0
    *combined with* poor EV means it has collapsed to always-hit instead, which
    is why the spec baseline is judged on ``max(rate, 1 - rate)`` rather than on
    the rate alone.
    """
    states = [
        (player_sum, upcard, ace)
        for player_sum in HITTABLE_SUMS
        for upcard in range(1, 11)
        for ace in (0, 1)
        # A usable ace requires a total of at least 12 (ace counted as 11).
        if not (ace == 1 and player_sum < 12)
    ]
    features = np.stack([preprocess_state(s) for s in states])
    actions = brain.policy(features).argmax(axis=1)
    return float((actions == 0).mean())


def train_spec_baseline(
    episodes: int = 200_000,
    seed: int = 0,
    lr: float = 0.005,
    checkpoint_every: int = 0,
    verbose: bool = False,
) -> tuple[FlyBlackjackBrain, TrainingHistory]:
    """The original specification's training loop, reproduced faithfully.

    One environment step per episode, ``loss = -log_prob * reward``, immediate
    reset. Preserved so its failure is *measured* against the corrected trainer
    rather than merely asserted. The only deviation from the spec listing is the
    backend (hand-written NumPy instead of PyTorch); the update rule, learning
    rate, optimizer and architecture are identical.
    """
    rng = np.random.default_rng(seed)
    brain = FlyBlackjackBrain(rng=np.random.default_rng(seed), lr=lr)
    env = _make_env()
    history = TrainingHistory(
        config={
            "trainer": "spec_baseline",
            "episodes": episodes,
            "seed": seed,
            "lr": lr,
        }
    )

    state, _ = env.reset(seed=seed)
    for episode in range(episodes):
        features = preprocess_state(state)[None, :]

        logits = brain.forward(features)
        probs = _softmax(logits)
        action = int(rng.choice(2, p=probs[0]))

        _, reward, _terminated, _truncated, _ = env.step(action)

        # The spec's update: scaled by the immediate reward, which is 0.0 for
        # any hit that does not bust, so most HIT experiences vanish entirely.
        dlogits = _policy_gradient_dlogits(
            probs, np.array([action]), np.array([float(reward)]), entropy_coef=0.0
        )
        brain.backward(dlogits)
        brain.step()

        history.episode_rewards.append(float(reward))

        # The spec resets every episode regardless of termination.
        state, _ = env.reset()

        if checkpoint_every and (episode + 1) % checkpoint_every == 0:
            history.checkpoints.append(
                {
                    "episode": episode + 1,
                    "mean_reward_recent": float(
                        np.mean(history.episode_rewards[-checkpoint_every:])
                    ),
                    "stand_rate_on_hittable": stand_rate_on_hittable(brain),
                }
            )
            if verbose:
                last = history.checkpoints[-1]
                print(
                    f"  [spec_baseline] episode {last['episode']}/{episodes} "
                    f"reward={last['mean_reward_recent']:+.4f} "
                    f"stand_rate={last['stand_rate_on_hittable']:.3f}"
                )

    env.close()
    return brain, history


def train_reinforce(
    episodes: int = 200_000,
    seed: int = 0,
    lr: float = TUNED_HYPERPARAMETERS["lr"],
    gamma: float = 1.0,
    baseline: bool = True,
    entropy_coef: float = TUNED_HYPERPARAMETERS["entropy_coef"],
    batch_episodes: int = 1,
    lr_final: float | None = TUNED_HYPERPARAMETERS["lr_final"],
    encoder=preprocess_state,
    brain: FlyBlackjackBrain | None = None,
    checkpoint_every: int = 0,
    verbose: bool = False,
) -> tuple[FlyBlackjackBrain, TrainingHistory]:
    """Corrected REINFORCE: full hands, return-to-go, baseline, entropy bonus.

    Each hand is played to termination and every decision in it is credited with
    the return that followed it, so a hit that sets up a winning stand finally
    receives the dopaminergic pulse it earned. ``gamma`` defaults to 1.0 because
    a blackjack hand is short and undiscounted return is the natural objective.

    The running-mean baseline subtracts the average return seen so far, which
    cuts gradient variance without biasing the estimator. Gradients are
    accumulated across ``batch_episodes`` hands and averaged before a single
    Adam step, which stabilises training at the spec's fairly high 0.005
    learning rate.

    ``encoder`` and ``brain`` exist for the sensory-encoding ablation: pass a
    wider encoder together with a brain built for that input width. The default
    is the spec's verbatim 3-feature mapping into the spec's 4,010-parameter
    network.
    """
    rng = np.random.default_rng(seed)
    if brain is None:
        brain = FlyBlackjackBrain(rng=np.random.default_rng(seed), lr=lr)

    probe_width = int(encoder((12, 5, 0)).shape[0])
    if probe_width != brain.LAYER_SIZES[0]:
        raise ValueError(
            f"encoder produces {probe_width} features but the brain expects "
            f"{brain.LAYER_SIZES[0]}; the rich-encoding ablation must pass a "
            "brain built for that input width"
        )

    env = _make_env()
    history = TrainingHistory(
        config={
            "trainer": "reinforce",
            "episodes": episodes,
            "seed": seed,
            "lr": lr,
            "gamma": gamma,
            "baseline": baseline,
            "entropy_coef": entropy_coef,
            "batch_episodes": batch_episodes,
            "lr_final": lr_final,
        }
    )

    baseline_sum = 0.0
    baseline_count = 0
    pending = 0
    env.reset(seed=seed)

    for episode in range(episodes):
        if lr_final is not None and episodes > 1:
            # Linear decay: large steps to find the policy, small ones to settle
            # on it. Without this the policy keeps jittering around the optimum
            # because every Adam step moves each weight by roughly lr.
            progress = episode / (episodes - 1)
            brain.lr = lr + (lr_final - lr) * progress

        state, _ = env.reset()
        step_features: list[np.ndarray] = []
        step_actions: list[int] = []
        step_rewards: list[float] = []

        for _ in range(MAX_STEPS_PER_HAND):
            features = encoder(state)
            probs = brain.policy(features[None, :])[0]
            action = int(rng.choice(2, p=probs))

            state, reward, terminated, truncated, _ = env.step(action)

            step_features.append(features)
            step_actions.append(action)
            step_rewards.append(float(reward))

            if terminated or truncated:
                break
        else:  # pragma: no cover - only reachable if the env stops terminating
            raise RuntimeError(
                f"hand exceeded {MAX_STEPS_PER_HAND} decisions; environment "
                "is not terminating as expected"
            )

        # Return-to-go: credit every decision with everything that followed it.
        returns = np.empty(len(step_rewards), dtype=np.float64)
        running = 0.0
        for t in range(len(step_rewards) - 1, -1, -1):
            running = step_rewards[t] + gamma * running
            returns[t] = running

        if baseline:
            mean_return = baseline_sum / baseline_count if baseline_count else 0.0
            advantages = returns - mean_return
            baseline_sum += float(returns.sum())
            baseline_count += len(returns)
        else:
            advantages = returns

        # Parameters are unchanged during the hand, so replaying every decision
        # in one batched forward is exactly equivalent to per-step passes.
        features_batch = np.stack(step_features)
        logits = brain.forward(features_batch)
        probs_batch = _softmax(logits)
        dlogits = _policy_gradient_dlogits(
            probs_batch,
            np.asarray(step_actions),
            advantages,
            entropy_coef=entropy_coef,
        )
        brain.backward(dlogits / batch_episodes)

        pending += 1
        if pending == batch_episodes:
            brain.step()
            pending = 0

        history.episode_rewards.append(float(step_rewards[-1]))

        if checkpoint_every and (episode + 1) % checkpoint_every == 0:
            history.checkpoints.append(
                {
                    "episode": episode + 1,
                    "mean_reward_recent": float(
                        np.mean(history.episode_rewards[-checkpoint_every:])
                    ),
                    "stand_rate_on_hittable": stand_rate_on_hittable(brain),
                }
            )
            if verbose:
                last = history.checkpoints[-1]
                print(
                    f"  [reinforce] episode {last['episode']}/{episodes} "
                    f"reward={last['mean_reward_recent']:+.4f} "
                    f"stand_rate={last['stand_rate_on_hittable']:.3f}"
                )

    if pending:  # flush a partial batch so no experience is silently dropped
        brain.step()

    env.close()
    return brain, history
