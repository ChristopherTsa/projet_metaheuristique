import numpy as np
from numba import njit

@njit
def greedy_heuristic(N, M, profits, resource_consumption, resource_availabilities):
    """
    Implements a greedy heuristic for the multidimensional knapsack problem using Numba for optimization.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        profits (np.ndarray): Array of profits for each project.
        resource_consumption (np.ndarray): 2D array of resource consumption (M x N).
        resource_availabilities (np.ndarray): Array of available quantities for each resource.

    Returns:
        np.ndarray: A feasible solution vector (array of 0s and 1s).
        float: Total profit of the solution.
    """
    # Step 1: Calculate profit-to-resource ratios
    ratios = np.zeros(N)
    for j in range(N):
        total_resource = np.sum(resource_consumption[:, j])
        if total_resource > 0:
            ratios[j] = profits[j] / total_resource
        else:
            ratios[j] = np.inf  # Projects with no resource consumption get the highest priority

    # Step 2: Sort projects by descending profit-to-resource ratio
    sorted_projects = np.argsort(ratios)[::-1]  # Indices sorted in descending order

    # Step 3: Initialize solution and resource usage
    solution = np.zeros(N, dtype=np.int32)
    used_resources = np.zeros(M, dtype=np.float64)

    # Step 4: Add projects to the solution while respecting resource constraints
    for j in sorted_projects:
        fits = True
        for i in range(M):
            if used_resources[i] + resource_consumption[i, j] > resource_availabilities[i]:
                fits = False
                break
        if fits:
            solution[j] = 1
            for i in range(M):
                used_resources[i] += resource_consumption[i, j]

    # Step 5: Calculate total profit
    total_profit = np.sum(profits * solution)

    return solution, total_profit


# Example usage:
# Assuming data is loaded from the previous function:
# data = read_mknap_data("mknap1.txt")
# instance = data[0]  # Use the first instance as an example
# solution, total_profit = greedy_heuristic(instance['N'], instance['M'], instance['profits'], instance['resource_consumption'], instance['resource_availabilities'])
# print("Solution:", solution)
# print("Total Profit:", total_profit)