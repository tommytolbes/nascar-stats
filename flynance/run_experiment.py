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
                          solution: opt.OptimalSolution) -> tuple[str, dict, object]:
    """Ask whether the spec's 3-scalar sensory encoding is the binding constraint.

    The spec maps a dealer ace to 0.1 -- numerically the weakest upcard, while
    strategically it is the strongest -- so the network has to learn a
    non-monotonic boundary from a single scalar. This retrains an identical
    network whose only difference is a wider input layer fed a one-hot dealer
    upcard, and reports what that buys.
    """
    from flynance.brain import FlyBlackjackBrain
    from flynance.encoding import RICH_INPUT_SIZE, preprocess_state_rich

    class WideSensoryBrain(FlyBlackjackBrain):
        """Same mushroom body, wider sensory layer."""

        LAYER_SIZES = (RICH_INPUT_SIZE, 64, 56, 2)

    brain = WideSensoryBrain(rng=np.random.default_rng(seed), lr=TUNED["lr"])
    brain, _ = train_reinforce(
        episodes=episodes,
        seed=seed,
        encoder=preprocess_state_rich,
        brain=brain,
        entropy_coef=TUNED["entropy_coef"],
    )
    policy_fn = ev.greedy_policy_fn(brain, encoder=preprocess_state_rich)
    result = ev.evaluate(policy_fn, n_hands=eval_hands, seed=seed,
                         label="reinforce + one-hot upcard encoding")
    align = strat.alignment(policy_fn, solution)

    lines = [
        f"Parameters: {brain.num_parameters()} "
        f"(vs {FlyBlackjackBrain.N_PARAMETERS} for the spec encoding)",
        f"EV over {eval_hands:,} hands: {result.ev:+.4f} "
        f"[{result.ci95[0]:+.4f}, {result.ci95[1]:+.4f}]",
        f"Weighted alignment with the optimum: {align['weighted']:.1%} "
        f"(raw {align['raw']:.1%})",
    ]
    metrics = {
        "n_parameters": int(brain.num_parameters()),
        "ev": float(result.ev),
        "ci95": [float(result.ci95[0]), float(result.ci95[1])],
        "weighted_alignment": float(align["weighted"]),
        "raw_alignment": float(align["raw"]),
    }
    return "\n".join(lines), metrics, brain


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
        ablation_text, ablation_metrics, ablation_brain = run_encoding_ablation(
            episodes=episodes, seed=seed, eval_hands=eval_hands, solution=solution
        )
        metrics_holder["ablation_brain"] = ablation_brain
        emit(ablation_text)
        emit(f"\nFor comparison, the spec encoding reached "
             f"{fly_alignment['weighted']:.1%} weighted alignment.")
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
    return "\n".join(lines), metrics, fly_brain, metrics_holder.get("ablation_brain")


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

    report, metrics, fly_brain, ablation_brain = run(
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
        if ablation_brain is not None:
            onehot_path = args.save_model.with_name(
                args.save_model.stem + "_onehot" + args.save_model.suffix)
            ablation_brain.save(onehot_path)
            print(f"One-hot ablation brain saved to {onehot_path}")
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
