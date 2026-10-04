import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from environment import Environment
from agent import Agent

def evaluate(agent):
    env = Environment()
    state = env.reset()
    total_reward = 0
    positions = []

    for _ in range(50):
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward

        positions.append(tuple(env.agent_pos))
        if len(positions) > 10:
            positions.pop(0)
            if all(p == positions[0] for p in positions):
                break

        if done:
            break

    return total_reward

def crossover(w1, w2):
    return [np.where(np.random.rand(*a.shape) < 0.5, a, b) for a, b in zip(w1, w2)]

def mutate(weights, rate=0.2):
    return [w + np.random.normal(0, rate, size=w.shape) for w in weights]

def evolve(pop_size=10, generations=30):
    global best_agent
    population = [Agent() for _ in range(pop_size)]
    best_fitness_log = []

    for gen in range(1, generations + 1):
        print(f"\n ♦♦♦ Generation {gen} ♦♦♦")
        scores = [evaluate(agent) for agent in population]
        best_fitness = max(scores)
        print(f"Best fitness: {best_fitness}")
        best_fitness_log.append(best_fitness)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pop_size // 2]
        top = [population[i] for i in top_indices]
        best_agent = top[0]

        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.sample(top, 2)
            child = Agent()
            w = crossover(p1.get_weights(), p2.get_weights())
            child.set_weights(mutate(w))
            new_population.append(child)

        population = new_population

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations + 1), best_fitness_log, marker='o', linestyle='-', color='blue')
    plt.title("Best Fitness Over 30 Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_fitness_over_generations.png")
    print("\nSaved fitness graph to 'best_fitness_over_generations.png'")

def run_best_agent():
    if 'best_agent' in globals():
        with open("\n best_agent_weights.pkl", "wb") as f:
            pickle.dump(best_agent.get_weights(), f)
        print("\n \nSaved best agent weights to 'best_agent_weights.pkl'\n ")
    else:
        print("Error: No best agent found. Did you run evolve() first?")