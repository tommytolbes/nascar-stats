"""Watch the trained fly brain play Blackjack, hand by hand.

    python -m flynance.watch                 # 10 hands, narrated
    python -m flynance.watch --hands 25      # more hands
    python -m flynance.watch --seed 7        # a different shuffle

On first run this trains a brain (a few minutes) and saves it to
``flynance/results/fly_brain.npz``; later runs load that file instantly. Use
``--retrain`` to force a fresh one.

Every decision shows the network's actual output: the softmax probability it
assigned to the action it chose. A confident 97% HIT and a coin-flip 52% STAND
look very different, and the difference is where the policy is still unsure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from flynance.brain import FlyBlackjackBrain
from flynance.encoding import preprocess_state
from flynance.trainers import train_reinforce

DEFAULT_MODEL = Path(__file__).resolve().parent / "results" / "fly_brain.npz"
ACTION_NAMES = {0: "STAND", 1: "HIT"}
CARD_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


def card_name(card: int) -> str:
    """Render a card value the way a player would say it."""
    return CARD_NAMES.get(card, str(card))


def hand_total(cards: list[int]) -> tuple[int, bool]:
    """Best total for a hand, and whether an ace is being counted as 11."""
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def describe_hand(cards: list[int]) -> str:
    """e.g. '[A, 6] = 17 (soft)' or '[10, 5] = 15'."""
    total, soft = hand_total(cards)
    rendered = ", ".join(card_name(c) for c in cards)
    return f"[{rendered}] = {total}{' (soft)' if soft else ''}"


def load_or_train(model_path: Path, episodes: int, seed: int,
                  retrain: bool = False) -> FlyBlackjackBrain:
    """Load the saved brain, training and saving one if there isn't a usable file."""
    if model_path.exists() and not retrain:
        print(f"Loading trained fly brain from {model_path}")
        return FlyBlackjackBrain.load(model_path)

    print(f"No saved brain at {model_path} -- training one "
          f"({episodes:,} hands, this takes a few minutes)...")
    brain, _ = train_reinforce(episodes=episodes, seed=seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    brain.save(model_path)
    print(f"Trained and saved to {model_path}\n")
    return brain


def play_hand(env: gym.Env, brain: FlyBlackjackBrain, hand_number: int) -> float:
    """Play one hand with running commentary. Returns the reward."""
    state, _ = env.reset()
    table = env.unwrapped
    dealer_upcard = table.dealer[0]

    print(f"\nHand {hand_number}")
    print("-" * 52)
    print(f"  Dealer shows   {card_name(dealer_upcard)}")
    print(f"  Fly holds      {describe_hand(list(table.player))}")

    while True:
        probs = brain.policy(preprocess_state(state)[None, :])[0]
        action = int(probs.argmax())
        confidence = float(probs[action])

        print(f"  Fly decides    {ACTION_NAMES[action]:<5}  "
              f"({confidence:.0%} confident)")

        before = list(table.player)
        state, reward, terminated, truncated, _ = env.step(action)

        if action == 1:
            drawn = [c for c in table.player[len(before):]]
            for card in drawn:
                print(f"  Draws          {card_name(card)}  ->  "
                      f"{describe_hand(list(table.player))}")

        if terminated or truncated:
            break

    # The dealer only plays out when the fly stands; on a bust it never acts.
    player_total, _ = hand_total(list(table.player))
    if player_total > 21:
        print(f"  BUST at {player_total}")
    else:
        print(f"  Dealer plays   {describe_hand(list(table.dealer))}")
        dealer_total, _ = hand_total(list(table.dealer))
        if dealer_total > 21:
            print(f"  Dealer BUSTS at {dealer_total}")

    outcome = {1.0: "WIN ", 0.0: "PUSH", -1.0: "LOSS"}[float(reward)]
    print(f"  Result         {outcome}  {float(reward):+.0f}")
    return float(reward)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", type=int, default=10, help="hands to play (default 10)")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=200_000,
                        help="training hands, if a brain must be trained first")
    parser.add_argument("--retrain", action="store_true",
                        help="train a fresh brain even if a saved one exists")
    args = parser.parse_args()

    brain = load_or_train(args.model, args.episodes, args.seed, args.retrain)
    print(f"{brain.num_parameters():,} synaptic weights, "
          f"{brain.LAYER_SIZES[0]} -> {' -> '.join(str(n) for n in brain.LAYER_SIZES[1:])}")

    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=args.seed)

    rewards = []
    try:
        for hand in range(1, args.hands + 1):
            rewards.append(play_hand(env, brain, hand))
    finally:
        env.close()

    wins = sum(1 for r in rewards if r > 0)
    losses = sum(1 for r in rewards if r < 0)
    pushes = sum(1 for r in rewards if r == 0)
    total = sum(rewards)

    print("\n" + "=" * 52)
    print(f"  {args.hands} hands:  {wins} won, {losses} lost, {pushes} pushed")
    print(f"  Net: {total:+.0f} units  ({total / args.hands:+.3f} per hand)")
    print("=" * 52)
    print("\nOver 10,000 hands this policy averages -0.049 per hand;")
    print("a short run like this is mostly luck, so don't read much into it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
