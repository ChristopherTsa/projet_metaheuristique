import numpy as np

def greedy_heuristic(N, M, profits, resource_consumption, resource_availabilities):
    """
    Implements a greedy heuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        profits (list): List of profits for each project.
        resource_consumption (list of lists): Resource consumption matrix.
        resource_availabilities (list): List of available quantities for each resource.

    Returns:
        list: A feasible solution vector (list of 0s and 1s).
        int: Total profit of the solution.
    """
    # Calculate profit-to-resource ratios
    ratios = []
    for j in range(N):
        total_resource = sum(resource_consumption[i][j] for i in range(M))
        if total_resource > 0:
            ratios.append(profits[j] / total_resource)
        else:
            ratios.append(0)

    # Sort projects by descending ratio of profit-to-resource
    sorted_projects = sorted(range(N), key=lambda x: ratios[x], reverse=True)

    # Initialize solution and resource usage
    solution = [0] * N
    used_resources = [0] * M

    # Add projects to the solution if they fit within the resource constraints
    for j in sorted_projects:
        fits = all(used_resources[i] + resource_consumption[i][j] <= resource_availabilities[i] for i in range(M))
        if fits:
            solution[j] = 1
            for i in range(M):
                used_resources[i] += resource_consumption[i][j]

    # Calculate total profit
    total_profit = sum(profits[j] * solution[j] for j in range(N))

    return solution, total_profit

# Example usage:
# Assuming data is loaded from the previous function:
# data = read_mknap_data("mknap1.txt")
# instance = data[0]  # Use the first instance as an example
# solution, total_profit = greedy_knapsack(instance['N'], instance['M'], instance['profits'], instance['resource_consumption'], instance['resource_availabilities'])
# print("Solution:", solution)
# print("Total Profit:", total_profit)