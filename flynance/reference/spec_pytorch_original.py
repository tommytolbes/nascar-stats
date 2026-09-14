"""VERBATIM copy of the PyTorch listing from the original project specification.

This file is PRESERVED FOR REFERENCE ONLY and is never imported or executed by
the flynance package. It is kept so the original proposal stays auditable next
to the corrected implementation.

Known defects in this listing (see flynance/README.md "Spec errata"):

  1. The training loop steps the environment exactly once per episode, then
     resets. A HIT that does not bust returns reward=0.0, so the loss is
     identically zero and no gradient flows. HIT therefore only ever receives
     the -1.0 bust signal while STAND receives an immediate +/-1 every episode,
     driving the policy toward always-stand. States reached after a hit are
     never visited.
  2. `terminated` and `truncated` are unpacked and then discarded.
  3. The spec's stated parameter count (4,180 / "~4,184") is wrong: the
     architecture below has 3*64+64 + 64*56+56 + 56*2+2 = 4,010 parameters.
  4. No evaluation or decision-matrix code is provided, although the spec's
     section 6 requires both.

The corrected implementation lives in flynance/brain.py and flynance/trainers.py.
The spec's loop itself is reproduced faithfully in trainers.train_spec_baseline()
so its failure mode can be measured rather than merely asserted.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# Define the micro-scale Fly Brain Network
class FlyBlackjackBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Sensory Layer (Input: 3 -> 64)
        self.sensory_layer = nn.Linear(3, 64)
        # Mushroom Body / Central Complex Sub-Network (64 -> 56)
        self.mushroom_body = nn.Linear(64, 56)
        # Motor Output Layer (56 -> 2 actions: 0=Stand, 1=Hit)
        self.motor_layer = nn.Linear(56, 2)

    def forward(self, x):
        x = torch.relu(self.sensory_layer(x))
        x = torch.relu(self.mushroom_body(x))
        return self.motor_layer(x)


# Preprocessing helper function
def preprocess_state(state):
    player_sum, dealer_card, usable_ace = state
    return torch.tensor([
        player_sum / 21.0,
        dealer_card / 10.0,
        float(usable_ace)
    ], dtype=torch.float32)


# Training loop initialization
def train_fly_agent(num_episodes=5000):
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    fly_brain = FlyBlackjackBrain()
    optimizer = optim.Adam(fly_brain.parameters(), lr=0.005)

    print(f"Starting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_tensor = preprocess_state(state)

        # Policy evaluation
        logits = fly_brain(state_tensor)
        probs = torch.softmax(logits, dim=-1)

        # Action selection
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        # Policy gradient update (Dopaminergic / Octopaminergic reward update)
        loss = -action_dist.log_prob(action) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed successfully.")

    env.close()
    return fly_brain


if __name__ == "__main__":
    trained_brain = train_fly_agent(5000)
