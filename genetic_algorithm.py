import json
import numpy as np
from tabulate import tabulate
import time

# Load the dataset
with open('data/data.json', 'r') as f:
    dataset = json.load(f)


def genetic_algorithm(capacity, weights, profits, generations=500, population_size=500, mutation_rate=0.1, convergence_threshold=100):
    
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

    population = []
    while len(population) < population_size:
        individual = list(np.random.randint(0, 2, n))
        if sum(weights[i] for i in range(n) if individual[i] == 1) <= capacity:
            population.append(individual)

    best_solution = population[0]
    best_fitness = fitness(best_solution)
    consecutive_no_improvement = 0

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

        new_best = parents[0]
        new_fitness = fitness(new_best)

        if new_fitness > best_fitness:
            best_solution = new_best
            best_fitness = new_fitness
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= convergence_threshold:
            break

    selected_items = [i for i in range(n) if best_solution[i] == 1]

    # Convert the solution to binary format
    binary_solution = [1 if i in selected_items else 0 for i in range(n)]

    
    return best_fitness, sum(weights[i] for i in selected_items), sum(profits[i] for i in selected_items), binary_solution

results_ga = []
    

for key, problem in dataset.items():
    
    start_time= time.time()
    
    capacity = problem['capacity'][0]
    weights = problem['weights']
    profits = problem['profits']
    optimal = problem['optimal']


    ga_profit, ga_weight, ga_optimal_profit, ga_selected_items = genetic_algorithm(capacity, weights, profits)
    ga_matches_optimal = ga_selected_items == optimal

    end_time= time.time()
    ga_runtime= end_time-start_time

    results_ga.append([key, "Genetic Algorithm", capacity, ga_profit, ga_weight, ga_optimal_profit, ga_selected_items, ga_matches_optimal, ga_runtime])



headers = ["Problem", "Algorithm", "Capacity", "Profit", "Weight", "Optimal Value", "Solution", "Matches Optimal", "Runtime"]
table = tabulate(results_ga, headers, tablefmt="grid")
print(table)

with open("results_ga.txt", "w") as f:
    f.write(table)