import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from environment import Environment
from agent import Agent

def plot_agent_path(steps, resource_positions, grid_size=5):
    grid = np.full((grid_size, grid_size), "", dtype=object)

    for i, (_, _, _, pos, _) in enumerate(steps):
        x, y = pos
        grid[x][y] = f"S{i+1}"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()
    ax.grid(True)

    for i in range(grid_size):
        for j in range(grid_size):
            label = grid[i, j]
            if label:
                ax.text(j, i, label, ha='center', va='center', fontsize=9, color='orange')

    for res in resource_positions:
        ax.add_patch(plt.Circle((res[1], res[0]), 0.3, color='blue', alpha=0.6))

    start_pos = steps[0][1]
    end_pos = steps[-1][3]
    ax.add_patch(plt.Circle((start_pos[1], start_pos[0]), 0.25, color='green', alpha=0.6, label='Start'))
    ax.add_patch(plt.Circle((end_pos[1], end_pos[0]), 0.25, color='red', alpha=0.6, label='End'))

    ax.set_title("Loaded Agent Path")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Resource', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='End', markerfacecolor='red', markersize=10),
    ])
    plt.tight_layout()
    plt.savefig("agent_path_from_saved_weights.png")
    print("Saved test run plot to 'agent_path_from_saved_weights.png'")
    plt.close()

def run_loaded_agent():
    with open("best_agent_weights.pkl", "rb") as f:
        weights = pickle.load(f)

    agent = Agent()
    agent.set_weights(weights)

    env = Environment()
    state = env.reset()
    total_reward = 0
    steps = []
    last_positions = []

    print("\n*** Loaded Agent Test Run ***")
    for step_num in range(30):
        action = agent.act(state)
        prev_pos = env.agent_pos[:]
        state, reward, done = env.step(action)
        total_reward += reward
        steps.append((step_num + 1, prev_pos, action, env.agent_pos[:], reward))

        last_positions.append(tuple(env.agent_pos))
        if len(last_positions) > 10:
            last_positions.pop(0)
            if all(p == last_positions[0] for p in last_positions):
                print("Oops!! Agent stuck. Stopping early.")
                break

        if len(env.collected) == len(env.resources):
            print("All resources collected. YAYY!! Ending early.")
            break

    print(f"\nTotal Reward Collected: {total_reward}")
    print("\nHere is step-by-step log:")
    for s in steps:
        print(f"Step {s[0]}: From {s[1]} -> Action {s[2]} -> To {s[3]} | Reward: {s[4]}")

    plot_agent_path(steps, env.resources)

    with open("best_agent_step_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "From", "Action", "To", "Reward"])
        for step in steps:
            writer.writerow([step[0], step[1], step[2], step[3], step[4]])

    print("Saved step-by-step log to 'best_agent_step_log.csv'")

if __name__ == "__main__":
    run_loaded_agent()
