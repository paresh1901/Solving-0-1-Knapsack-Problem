import json
import numpy as np
from tabulate import tabulate
import time

# Load the dataset
with open('data/data.json', 'r') as f:
    dataset = json.load(f)

def dynamic_programming(capacity, weights, profits):
    
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

    
    return dp[n][capacity], sum(weights[i] for i in selected_items), sum(profits[i] for i in selected_items), binary_solution 

results_dp = []  # Initialize the results list
repetitions = 200  # Number of repetitions for timing

for key, problem in dataset.items():
    
    total_runtime = 0
    
    capacity = problem['capacity'][0]
    weights = problem['weights']
    profits = problem['profits']
    optimal = problem['optimal']

    for _ in range(repetitions):
        start_time = time.process_time()  # Use time.process_time() for measuring CPU time
        dp_profit, dp_weight, dp_optimal_profit, dp_selected_items = dynamic_programming(capacity, weights, profits)
        end_time = time.process_time()
        total_runtime += (end_time - start_time)

    dp_runtime = total_runtime / repetitions

    dp_matches_optimal = dp_selected_items == optimal

    results_dp.append([key, "Dynamic Programming", capacity, dp_profit, dp_weight, dp_optimal_profit, dp_selected_items, dp_matches_optimal, f"{dp_runtime:.6f} s"])

# Create and print the table
headers = ["Problem", "Algorithm", "Capacity", "Profit", "Weight", "Optimal Value", "Solution", "Matches Optimal", "Runtime"]
table = tabulate(results_dp, headers, tablefmt="grid")
print(table)

with open("results_dp.txt", "w") as f:
    f.write(table)


# Create a dictionary to store the results
results_dict = []

for result in results_dp:
    result_dict = {
        "Problem": result[0],
        "Algorithm": result[1],
        "Capacity": result[2],
        "Profit": result[3],
        "Weight": result[4],
        "Optimal Value": result[5],
        "Solution": result[6],
        "Matches Optimal": result[7],
        "Runtime": result[8]
    }
    results_dict.append(result_dict)

# Save the results as a JSON file
with open("results_dp.json", "w") as json_file:
    json.dump(results_dict, json_file, indent=4)

