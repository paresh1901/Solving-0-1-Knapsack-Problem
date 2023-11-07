import json
import matplotlib.pyplot as plt

# Load results from JSON files
with open("results_dp.json", "r") as f:
    results_dp = json.load(f)

with open("results_ga.json", "r") as f:
    results_ga = json.load(f)

# Extract relevant data for comparison
problem_names = [result["Problem"] for result in results_dp]
dp_runtimes = [result["Runtime"] for result in results_dp]
ga_runtimes = [result["Runtime"] for result in results_ga]

# Convert runtime strings to float values
dp_runtimes = [float(runtime.split()[0]) for runtime in dp_runtimes]
ga_runtimes = [float(runtime) for runtime in ga_runtimes]

# Create a line plot for runtime comparison
plt.figure(figsize=(12, 8))  # Increased figure size

# First subplot for runtime comparison
plt.subplot(211)
plt.plot(problem_names, dp_runtimes, marker='o', label='Dynamic Programming', color='b', linestyle='-', markersize=6)
plt.plot(problem_names, ga_runtimes, marker='o', label='Genetic Algorithm', color='g', linestyle='-', markersize=6)

plt.xlabel('Problems')
plt.ylabel('Runtime (s)')
plt.title('Algorithm Runtimes for Knapsack Problem')
plt.xticks(rotation=45)
plt.grid()
plt.legend()

# Extract relevant data for optimal solution matches comparison
dp_matches_optimal = [result["Matches Optimal"] for result in results_dp]
ga_matches_optimal = [result["Matches Optimal"] for result in results_ga]

# Create a scatter plot to compare optimal solution matches
plt.subplot(212)

# Scatter points for Dynamic Programming
dp_x = [i for i, match in enumerate(dp_matches_optimal) if match]
dp_y = [1] * len(dp_x)  # Assign y=1 for matching results

# Scatter points for Genetic Algorithm
ga_x = [i for i, match in enumerate(ga_matches_optimal) if match]
ga_y = [2] * len(ga_x)  # Assign y=2 for matching results

# Scatter points for non-matching results
dp_non_match_x = [i for i, match in enumerate(dp_matches_optimal) if not match]
dp_non_match_y = [1] * len(dp_non_match_x)  # Assign y=1 for non-matching results
ga_non_match_x = [i for i, match in enumerate(ga_matches_optimal) if not match]
ga_non_match_y = [2] * len(ga_non_match_x)  # Assign y=2 for non-matching results

# Increase the size of the dots and icons
dot_size = 150

# Plot scatter points
plt.scatter(dp_x, dp_y, label='Dynamic Programming Matches', color='b', marker='o', s=dot_size)
plt.scatter(ga_x, ga_y, label='Genetic Algorithm Matches', color='g', marker='o', s=dot_size)
plt.scatter(dp_non_match_x, dp_non_match_y, label='Dynamic Programming Non-Matches', color='r', marker='x', s=dot_size)
plt.scatter(ga_non_match_x, ga_non_match_y, label='Genetic Algorithm Non-Matches', color='m', marker='x', s=dot_size)

plt.yticks([1, 2], ['Dynamic Programming', 'Genetic Algorithm'])
plt.xlabel('Problems')
plt.title('Optimal Solution Matches for Knapsack Problem')

# Place the legend in the center of the graph
plt.legend(loc='center')

plt.tight_layout()

# Save the combined graph as an image file
plt.savefig('runtime_and_matches_comparison.png')

# Show the graph
plt.show()
