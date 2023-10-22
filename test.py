import json
import numpy as np
from tabulate import tabulate
import time

# Load the dataset
with open('data/data.json', 'r') as f:
    dataset = json.load(f)

def dynamic_programming(capacity, weights, profits):
    start_time = time.time()
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(profits[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()

    # Convert the solution to binary format
    binary_solution = [1 if i in selected_items else 0 for i in range(n)]

    end_time = time.time()
    return dp[n][capacity], sum(weights[i] for i in selected_items), sum(profits[i] for i in selected_items), binary_solution, end_time - start_time

def genetic_algorithm(capacity, weights, profits, generations=1000, population_size=1000, mutation_rate=0.1):
    start_time = time.time()
    n = len(weights)

    def fitness(chromosome):
        total_weight = sum(weights[i] for i in range(n) if chromosome[i] == 1)
        total_profit = sum(profits[i] for i in range(n) if chromosome[i] == 1)
        if total_weight > capacity:
            return 0
        return total_profit

    def mutate(chromosome):
        for i in range(n):
            if np.random.rand() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]

    population = [list(np.random.randint(0, 2, n)) for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=lambda x: -fitness(x))
        parents = population[:population_size // 2]

        for _ in range(population_size // 2):
            indices = np.random.choice(range(len(parents)), 2, replace=False)
            parent1 = parents[indices[0]]
            parent2 = parents[indices[1]]
            crossover_point = np.random.randint(1, n)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            mutate(child)
            parents.append(child)

    best_solution = parents[0]
    selected_items = [i for i in range(n) if best_solution[i] == 1]

    # Convert the solution to binary format
    binary_solution = [1 if i in selected_items else 0 for i in range(n)]

    end_time = time.time()
    return fitness(best_solution), sum(weights[i] for i in selected_items), sum(profits[i] for i in selected_items), binary_solution, end_time - start_time

results = []

for key, problem in dataset.items():
    capacity = problem['capacity'][0]
    weights = problem['weights']
    profits = problem['profits']
    optimal = problem['optimal']

    dp_profit, dp_weight, dp_optimal_profit, dp_selected_items, dp_runtime = dynamic_programming(capacity, weights, profits)
    dp_matches_optimal = dp_selected_items == optimal

    ga_profit, ga_weight, ga_optimal_profit, ga_selected_items, ga_runtime = genetic_algorithm(capacity, weights, profits)
    ga_matches_optimal = ga_selected_items == optimal

    results.append([key, "Dynamic Programming", capacity, dp_profit, dp_weight, dp_optimal_profit, dp_selected_items, optimal, dp_matches_optimal, dp_runtime])
    results.append([key, "Genetic Algorithm", capacity, ga_profit, ga_weight, ga_optimal_profit, ga_selected_items, optimal, ga_matches_optimal, ga_runtime])

# Create and print the table
headers = ["Problem", "Algorithm", "Capacity", "Profit", "Weight", "Optimal Value", "Solution", "Actual Solution", "Matches Optimal", "Runtime"]
table = tabulate(results, headers, tablefmt="grid")
print(table)
