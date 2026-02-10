"""Compare a trained LunarLander policy against a random policy headlessly."""

from pathlib import Path
from statistics import fmean
from typing import Callable, List

import matplotlib.pyplot as plt
import torch

from lunar_lander_dqn import DQN, create_env, device

# Ensure blocking show() despite lunar_lander_dqn enabling interactive mode
plt.ioff()


ActionFn = Callable[[torch.Tensor], int]

WEIGHTS_PATH = Path("artifacts/lunar_lander_dqn.pt")
NUM_EPISODES = 100


def load_policy(env, weights_path: Path) -> DQN:
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy = DQN(n_observations, n_actions).to(device)
    checkpoint = torch.load(weights_path.expanduser().resolve(), map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    return policy


def run_episode(env, select_action: Callable[[torch.Tensor], int]) -> float:
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def greedy_action_fn(policy: DQN) -> Callable[[torch.Tensor], int]:
    def select_action(state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return policy(state_tensor).max(1).indices.item()

    return select_action


def random_action_fn(env) -> Callable[[torch.Tensor], int]:
    return lambda _state: env.action_space.sample()


def evaluate_policy(env, select_action: Callable[[torch.Tensor], int], episodes: int) -> List[float]:
    return [run_episode(env, select_action) for _ in range(episodes)]


def plot_results(policy_mean: float, random_mean: float, episodes: int):
    labels = ["DQN Policy", "Random Policy"]
    means = [policy_mean, random_mean]
    colors = ["#4C72B0", "#DD8452"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, means, color=colors)
    plt.ylabel("Average Reward")
    plt.title(f"LunarLander average reward over {episodes} episodes")

    for bar, value in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center",
                 va="bottom" if value >= 0 else "top")

    plt.tight_layout()
    plt.show()
    # plt.close()


def main():
    # Evaluate trained policy
    policy_env = create_env()
    policy = load_policy(policy_env, WEIGHTS_PATH)
    trained_rewards = evaluate_policy(policy_env, greedy_action_fn(policy), NUM_EPISODES)
    policy_env.close()

    # Evaluate random policy
    random_env = create_env()
    random_rewards = evaluate_policy(random_env, random_action_fn(random_env), NUM_EPISODES)
    random_env.close()

    policy_mean = fmean(trained_rewards)
    random_mean = fmean(random_rewards)

    print(f"Trained policy average reward over {NUM_EPISODES} episodes: {policy_mean:.2f}")
    print(f"Random policy average reward over {NUM_EPISODES} episodes: {random_mean:.2f}")

    plot_results(policy_mean, random_mean, NUM_EPISODES)


if __name__ == "__main__":
    main()
