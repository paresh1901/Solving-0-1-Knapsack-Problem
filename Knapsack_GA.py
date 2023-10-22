import random
import matplotlib.pyplot as plt

# Increased data complexity and size
weights = [12, 25, 35, 55, 45, 60, 30, 22, 18, 50, 42, 38, 48, 28, 32, 56, 40, 36, 29, 44]
values = [72, 110, 125, 200, 180, 240, 160, 80, 70, 220, 190, 150, 210, 140, 170, 210, 175, 160, 95, 140]

knapsack_capacity = 300
population_size = 100
generations = 50
crossover_rate = 0.85
mutation_rate = 0.01

# Initialize the population
def initialize_population(size):
    population = []
    for _ in range(size):
        individual = [random.randint(0, 1) for _ in range(len(weights))]
        population.append(individual)
    return population

# Calculate the fitness of an individual
def fitness(individual):
    total_weight = sum(w * x for w, x in zip(weights, individual))
    total_value = sum(v * x for v, x in zip(values, individual))
    if total_weight > knapsack_capacity:
        return 0  # Penalize solutions that exceed the capacity
    return total_value

# Select parents using roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [fit / total_fitness for fit in fitness_values]
    selected = random.choices(population, weights=selection_probs, k=2)
    return selected

# Perform single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutate an individual
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# Main genetic algorithm with added tracking
def genetic_algorithm_with_tracking():
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = 0
    best_fitness_over_generations = []
    average_fitness_over_generations = []
    population_diversity_over_generations = []

    for generation in range(generations):
        fitness_values = [fitness(ind) for ind in population]
        max_fitness = max(fitness_values)
        best_fitness_over_generations.append(max_fitness)

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitness_values.index(max_fitness)]

        if fitness_values.count(max_fitness) / population_size >= 0.95:
            break

        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = roulette_wheel_selection(population, fitness_values)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

        # Calculate population diversity
        population_diversity = len(set(map(tuple, population)))
        population_diversity_over_generations.append(population_diversity)

        # Calculate average fitness
        average_fitness = sum(fitness_values) / len(fitness_values)
        average_fitness_over_generations.append(average_fitness)

    return best_solution, best_fitness, best_fitness_over_generations, average_fitness_over_generations, population_diversity_over_generations

best_solution, best_fitness, best_fitness_over_generations, average_fitness_over_generations, population_diversity_over_generations = genetic_algorithm_with_tracking()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
print("Total weight in the knapsack:", sum(w * x for w, x in zip(weights, best_solution)))

# Plot the best fitness values over generations
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(best_fitness_over_generations, label="Best Fitness")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.title("Best Fitness Over Generations")
plt.grid(True)

# Plot the average fitness values over generations
plt.subplot(2, 2, 2)
plt.plot(average_fitness_over_generations, label="Average Fitness")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.title("Average Fitness Over Generations")
plt.grid(True)

# Plot the best fitness vs. average fitness over generations
plt.subplot(2, 2, 3)
plt.plot(best_fitness_over_generations, label="Best Fitness")
plt.plot(average_fitness_over_generations, label="Average Fitness")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.title("Best vs. Average Fitness Over Generations")
plt.grid(True)

# Plot population diversity over generations
plt.subplot(2, 2, 4)
plt.plot(population_diversity_over_generations, label="Population Diversity")
plt.xlabel("Generations")
plt.ylabel("Number of Unique Individuals")
plt.legend()
plt.title("Population Diversity Over Generations")
plt.grid(True)

plt.tight_layout()
plt.show()