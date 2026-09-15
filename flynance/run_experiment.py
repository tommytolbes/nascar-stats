"""Run the whole Agentic Flynance experiment and write the report.

    python -m flynance.run_experiment --episodes 200000 --seed 42

Trains both the spec-faithful baseline and the corrected REINFORCE agent,
evaluates every policy over the same seeded hands, scores the learned policy
against the exact dynamic-programming optimum, and writes everything to
``flynance/results/report.txt``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flynance import evaluate as ev
from flynance import optimal as opt
from flynance import strategy as strat
from flynance.trainers import (
    TUNED_HYPERPARAMETERS,
    stand_rate_on_hittable,
    train_reinforce,
    train_spec_baseline,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The corrected trainer's tuned defaults live in trainers.py; imported here so
# the report can state exactly what was used without duplicating the values.
TUNED = TUNED_HYPERPARAMETERS


def optimal_policy_fn(solution: opt.OptimalSolution) -> ev.PolicyFn:
    """Wrap the DP solution as a PolicyFn, for use as the performance ceiling."""

    def policy(state, rng):  # noqa: ARG001 - deterministic, rng unused
        action = solution.policy.get(state)
        if action is None:
            # Unreached corner (e.g. a total the deal cannot produce): fall back
            # to the dealer's own rule rather than guessing.
            return 0 if state[0] >= 17 else 1
        return int(action)

    return policy


def _section(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n"


def run_encoding_ablation(episodes: int, seed: int, eval_hands: int,
                          solution: opt.OptimalSolution) -> tuple[str, dict, dict]:
    """Walk the encoding ladder: how much of the fly's error is the input's fault?

    Three networks, identical but for how the state is presented to them:

    * the spec's three scalars, where a dealer ace arrives as 0.1 -- numerically
      the weakest upcard while being strategically the strongest;
    * a one-hot dealer upcard, which removes that false ordering;
    * everything one-hot, which also stops the player's total asserting that 16
      is "close to" 17 -- false exactly where the game's sharpest boundary sits.

    Each is scored by exact dynamic programming rather than by sampling, because
    the differences here are smaller than the noise in a 10,000-hand sweep.
    """
    from flynance.brain import FlyBlackjackBrain
    from flynance import encoding as enc

    variants = [
        ("one-hot upcard", enc.RICH_INPUT_SIZE, enc.preprocess_state_rich, "onehot"),
        ("full one-hot", enc.FULL_INPUT_SIZE, enc.preprocess_state_full_onehot, "full"),
    ]

    lines = [
        f"{'encoding':<18}{'params':>8}{'wrong':>7}{'weighted':>10}{'exact EV':>12}{'vs optimal':>12}",
        "-" * 67,
    ]
    metrics: dict = {}
    brains: dict = {}

    # The spec-encoded fly is trained by the main run; score it here for the ladder.
    for label, width, encoder, key in variants:
        class VariantBrain(FlyBlackjackBrain):
            LAYER_SIZES = (width, 64, 56, 2)

        brain = VariantBrain(rng=np.random.default_rng(seed), lr=TUNED["lr"])
        brain, _ = train_reinforce(
            episodes=episodes,
            seed=seed,
            encoder=encoder,
            brain=brain,
            entropy_coef=TUNED["entropy_coef"],
        )
        policy_fn = ev.greedy_policy_fn(brain, encoder=encoder)
        align = strat.alignment(policy_fn, solution)
        grid = {s: int(policy_fn(s, None)) for s in solution.policy}
        exact = opt.policy_expected_value(grid)

        lines.append(
            f"{label:<18}{brain.num_parameters():>8}{len(align['disagreements']):>7}"
            f"{align['weighted']:>9.1%}{exact:>12.6f}"
            f"{exact - solution.expected_value:>+12.6f}"
        )
        metrics[key] = {
            "label": label,
            "n_parameters": int(brain.num_parameters()),
            "wrong_cells": len(align["disagreements"]),
            "weighted_alignment": float(align["weighted"]),
            "exact_ev": float(exact),
        }
        brains[key] = brain

    return "\n".join(lines), metrics, brains


def run(
    episodes: int,
    seed: int,
    eval_hands: int,
    verbose: bool = True,
    ablation: bool = False,
) -> tuple[str, dict, object, object | None]:
    """Execute the full experiment.

    Returns the report text, the metrics dict, the trained REINFORCE brain, and
    the one-hot ablation brain (``None`` unless ``ablation`` was requested), so
    the caller can save them (see ``--save-model``) and replay them later.
    """
    started = time.time()
    metrics_holder: dict = {}
    lines: list[str] = []
    metrics: dict = {"config": {
        "episodes": episodes,
        "seed": seed,
        "eval_hands": eval_hands,
    }}

    def emit(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    emit("AGENTIC FLYNANCE - BLACKJACK FLY-BRAIN EXPERIMENT")
    emit(f"episodes={episodes}  seed={seed}  eval_hands={eval_hands}")

    # ---- 1. Exact ground truth --------------------------------------------
    emit(_section("1. EXACT OPTIMAL POLICY (dynamic programming)"))
    solution = opt.solve()
    emit(f"Exact expected value of optimal hit/stand play: {solution.expected_value:+.6f}")
    emit(f"Decision states solved: {len(solution.policy)}")
    metrics["optimal_ev"] = float(solution.expected_value)

    # ---- 2. Train both agents ---------------------------------------------
    emit(_section("2. TRAINING"))
    checkpoint_every = max(episodes // 10, 1)

    emit(f"Training spec-faithful baseline ({episodes} episodes)...")
    spec_brain, spec_history = train_spec_baseline(
        episodes=episodes, seed=seed, checkpoint_every=checkpoint_every, verbose=verbose
    )
    spec_stand_rate = stand_rate_on_hittable(spec_brain)
    collapse = ("always-stand" if spec_stand_rate > 0.9
                else "always-hit" if spec_stand_rate < 0.1
                else "mixed")
    emit(f"  parameters: {spec_brain.num_parameters()}")
    emit(f"  stand-rate on hittable states (4-16): {spec_stand_rate:.4f}  [{collapse}]")

    emit(f"\nTraining corrected REINFORCE agent ({episodes} episodes, "
         f"lr={TUNED['lr']} -> {TUNED['lr_final']}, entropy={TUNED['entropy_coef']})...")
    fly_brain, fly_history = train_reinforce(
        episodes=episodes,
        seed=seed,
        checkpoint_every=checkpoint_every,
        verbose=verbose,
    )
    fly_stand_rate = stand_rate_on_hittable(fly_brain)
    emit(f"  parameters: {fly_brain.num_parameters()}")
    emit(f"  stand-rate on hittable states (4-16): {fly_stand_rate:.4f}")

    metrics["spec_stand_rate_on_hittable"] = float(spec_stand_rate)
    metrics["reinforce_stand_rate_on_hittable"] = float(fly_stand_rate)
    metrics["n_parameters"] = int(fly_brain.num_parameters())

    # ---- 3. Evaluation sweep ----------------------------------------------
    emit(_section(f"3. EXPECTED VALUE OVER {eval_hands:,} HANDS (common random numbers)"))
    policies = {
        "reinforce (fly brain)": ev.greedy_policy_fn(fly_brain),
        "spec_baseline (fly brain)": ev.greedy_policy_fn(spec_brain),
        "always_stand": ev.always_stand,
        "dealer_mimic (hit < 17)": ev.dealer_mimic,
        "DP optimal (ceiling)": optimal_policy_fn(solution),
    }
    results = ev.compare(policies, n_hands=eval_hands, seed=seed)
    emit(ev.render_comparison(results))
    metrics["evaluation"] = {
        r.label: {
            "ev": float(r.ev),
            "stderr": float(r.stderr),
            "ci95": [float(r.ci95[0]), float(r.ci95[1])],
            "win_rate": float(r.win_rate),
            "loss_rate": float(r.loss_rate),
            "push_rate": float(r.push_rate),
        }
        for r in results
    }

    # Sampling cannot separate a good policy from a perfect one at 10,000 hands
    # (stderr ~0.01 against a total gap of ~0.005), so the same policies are also
    # evaluated exactly, by dynamic programming over their own decisions.
    emit("Exact expected value of the same policies (no sampling):")
    exact_rows = []
    for label, policy_fn in policies.items():
        grid = {s: int(policy_fn(s, None)) for s in solution.policy}
        exact_rows.append((label, opt.policy_expected_value(grid)))
    exact_rows.sort(key=lambda r: r[1], reverse=True)
    width = max(len(label) for label, _ in exact_rows)
    for label, value in exact_rows:
        gap = value - solution.expected_value
        emit(f"  {label:<{width}}  {value:+.6f}   "
             f"{'(optimal)' if abs(gap) < 1e-9 else f'{gap:+.6f} vs optimal'}")
    metrics["exact_ev"] = {label: float(value) for label, value in exact_rows}

    # ---- 4. Decision matrices and alignment -------------------------------
    emit(_section("4. DECISION MATRIX vs EXACT OPTIMUM"))
    fly_policy = ev.greedy_policy_fn(fly_brain)
    reference = strat.decision_matrix(optimal_policy_fn(solution))
    learned = strat.decision_matrix(fly_policy)

    emit(strat.render_matrix(learned, "Fly brain (REINFORCE) - hard/soft totals", reference=reference))
    emit(strat.render_matrix(reference, "Exact DP optimum - hard/soft totals"))

    fly_alignment = strat.alignment(fly_policy, solution)
    emit(strat.render_alignment(fly_alignment))
    metrics["alignment"] = {
        "raw": float(fly_alignment["raw"]),
        "weighted": float(fly_alignment["weighted"]),
        "n_disagreements": len(fly_alignment["disagreements"]),
    }

    spec_alignment = strat.alignment(ev.greedy_policy_fn(spec_brain), solution)
    emit(f"\nspec_baseline alignment for comparison: "
         f"raw={spec_alignment['raw']:.1%}  weighted={spec_alignment['weighted']:.1%}")
    metrics["spec_alignment"] = {
        "raw": float(spec_alignment["raw"]),
        "weighted": float(spec_alignment["weighted"]),
    }

    # ---- 5. Optional encoding ablation ------------------------------------
    if ablation:
        emit(_section("5. ABLATION: is the spec's sensory encoding the ceiling?"))
        ablation_text, ablation_metrics, ablation_brains = run_encoding_ablation(
            episodes=episodes, seed=seed, eval_hands=eval_hands, solution=solution
        )
        metrics_holder["ablation_brains"] = ablation_brains
        emit(ablation_text)
        emit(f"\nThe spec's own encoding, same trainer: "
             f"{fly_alignment['weighted']:.1%} weighted, "
             f"{len(fly_alignment['disagreements'])} cells wrong.")
        metrics["ablation"] = ablation_metrics

    # ---- 6. Verdict --------------------------------------------------------
    emit(_section("6. SUCCESS CRITERIA"))
    by_label = {r.label: r for r in results}
    fly_result = by_label["reinforce (fly brain)"]
    spec_result = by_label["spec_baseline (fly brain)"]
    stand_result = by_label["always_stand"]

    def beats(a, b) -> bool:
        """True when a's 95% CI lies entirely above b's."""
        return a.ci95[0] > b.ci95[1]

    checks = [
        ("S1  brain has exactly 4,010 parameters",
         fly_brain.num_parameters() == 4010),
        # The spec loop collapses onto one constant action; which one is
        # seed-dependent, so degeneracy is measured in either direction.
        ("S4  spec_baseline collapses to a constant action (stand-rate > 0.9 or < 0.1)",
         max(spec_stand_rate, 1.0 - spec_stand_rate) > 0.9),
        ("S5a reinforce beats always_stand (non-overlapping 95% CI)",
         beats(fly_result, stand_result)),
        ("S5b reinforce beats spec_baseline (non-overlapping 95% CI)",
         beats(fly_result, spec_result)),
        ("S6  weighted alignment with DP optimum >= 85%",
         fly_alignment["weighted"] >= 0.85),
    ]
    for name, passed in checks:
        emit(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    metrics["criteria"] = {name: bool(passed) for name, passed in checks}
    metrics["all_criteria_passed"] = all(passed for _, passed in checks)

    emit(f"\nCompleted in {time.time() - started:.1f}s")
    return "\n".join(lines), metrics, fly_brain, metrics_holder.get("ablation_brains")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200_000,
                        help="training episodes per agent (default: 200000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-hands", type=int, default=10_000,
                        help="hands in the evaluation sweep (spec section 6: 10000)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "report.txt")
    parser.add_argument("--save-model", type=Path, default=RESULTS_DIR / "fly_brain.npz",
                        help="where to write the trained brain (.npz) for replay")
    parser.add_argument("--check-reproducible", action="store_true",
                        help="compare metrics against the stored run for this seed")
    parser.add_argument("--ablation", action="store_true",
                        help="also train a wider-input network on a one-hot dealer "
                             "upcard, to test whether the spec's 3-scalar encoding "
                             "is what limits the agent")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report, metrics, fly_brain, ablation_brains = run(
        episodes=args.episodes,
        seed=args.seed,
        eval_hands=args.eval_hands,
        verbose=not args.quiet,
        ablation=args.ablation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)

    # Persist the trained brain so it can be replayed without retraining
    # (flynance/watch.py loads exactly this file).
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        fly_brain.save(args.save_model)
        print(f"Trained brain saved to {args.save_model}")
        for key, brain in (ablation_brains or {}).items():
            path = args.save_model.with_name(
                f"{args.save_model.stem}_{key}{args.save_model.suffix}")
            brain.save(path)
            print(f"Ablation brain ({key}) saved to {path}")
    metrics_path = args.output.parent / f"metrics_seed{args.seed}.json"

    if args.check_reproducible and metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        # Compare only sections both runs actually computed: a stored run made
        # with --ablation carries a section a plain run never produces, and that
        # absence is not a reproducibility failure.
        shared = set(previous) & set(metrics)
        skipped = sorted((set(previous) ^ set(metrics)))
        differing = sorted(key for key in shared if previous[key] != metrics[key])

        print(f"\nReproducibility check vs {metrics_path.name}: "
              f"{'IDENTICAL' if not differing else 'DIFFERENT'} "
              f"({len(shared)} sections compared)")
        if skipped:
            print(f"  not compared (present in only one run): {', '.join(skipped)}")
        if differing:
            for key in differing:
                print(f"  {key}:\n    stored: {previous[key]}\n    rerun:  {metrics[key]}")
            return 1
    else:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print(f"\nReport written to {args.output}")
    print(f"Metrics written to {metrics_path}")
    return 0 if metrics.get("all_criteria_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
