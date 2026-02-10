import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def evaluate_policy(Q, policy="greedy", episodes=5000):
    env = gym.make("Blackjack-v1")
    total_reward = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            if policy == "greedy":
                current_state = Q[obs[0], obs[1], obs[2], :]
                if current_state[0] == current_state[1]:
                    action = np.random.choice([0, 1]) # if both actions have the same value, choose randomly
                else:
                    action = int(np.argmax(Q[obs[0], obs[1], obs[2], :]))
            else:
                action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    env.close()
    return total_reward / episodes

if __name__ == "__main__":

    # Initialize Q-value function (table)
    Q = np.zeros((32, 11, 2, 2))  # (player_sum, dealer_showing, usable_ace, action)

    # Hyperparams
    learning_rate = 0.1
    discount_factor = 0.9


    num_episodes = 1000
    current_episode = 0
    eval_interval = 1
    eval_points = []
    greedy_rewards = []
    random_rewards = []

    while current_episode < num_episodes:

        env = gym.make("Blackjack-v1")
        obs, info = env.reset()
        done = False

        while not done:
            current_state = Q[obs[0], obs[1], obs[2], :]
            
            if current_state[0] == current_state[1]:
                best_action = np.random.choice([0, 1]) # if both actions have the same value, choose randomly
            else:
                best_action = np.argmin(current_state) # default value is stand, 0 (returns first index if tie)

            new_obs, immediate_reward, done, _, info = env.step(best_action)

            best_future_reward = np.max(Q[new_obs[0], new_obs[1], new_obs[2], :]) # the maximum reward obtainable from the new state we landed in

            Q[obs[0], obs[1], obs[2], best_action] = Q[obs[0], obs[1], obs[2], best_action] + learning_rate * (immediate_reward + discount_factor * best_future_reward - Q[obs[0], obs[1], obs[2], best_action])

        current_episode += 1

        if current_episode % eval_interval == 0 or current_episode <= 1:
            greedy_avg = evaluate_policy(Q, policy="greedy", episodes=400)
            random_avg = evaluate_policy(Q, policy="random", episodes=400)
            eval_points.append(current_episode)
            greedy_rewards.append(greedy_avg)
            random_rewards.append(random_avg)
            print(f"Episode {current_episode}: greedy avg reward {greedy_avg:.3f}, random {random_avg:.3f}")
    
    print(Q)

    if eval_points:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_points, greedy_rewards, label="Q-learning policy")
        plt.plot(eval_points, random_rewards, label="Random policy")
        plt.xlabel("Episodes")
        plt.ylabel("Average reward")
        plt.title("Policy evaluation over training")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

