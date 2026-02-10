"""Load a trained DQN policy and render it on the LunarLander environment."""

import argparse
from pathlib import Path

import torch

from lunar_lander_dqn import DQN, create_env, device


def parse_args():
    parser = argparse.ArgumentParser(description="Render a trained LunarLander policy in a headful gym window.")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/lunar_lander_dqn.pt"),
                        help="Path to the saved policy weights produced by lunar_lander_dqn.py")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes to render")
    return parser.parse_args()


def load_policy(env, weights_path: Path) -> DQN:
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy = DQN(n_observations, n_actions).to(device)
    checkpoint = torch.load(weights_path.expanduser().resolve(), map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    return policy


def run_episode(env, policy: DQN) -> float:
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy(state_tensor).max(1).indices.item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    return total_reward


def main():
    args = parse_args()

    env = create_env(render_mode="human")
    policy = load_policy(env, args.weights)

    for episode in range(args.episodes):
        reward = run_episode(env, policy)
        print(f"Episode {episode} reward: {reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
