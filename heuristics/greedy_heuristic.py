import numpy as np
from utilities import calculate_profit, is_feasible


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

    # Step 3: Initialize solution
    solution = np.zeros(N, dtype=np.int32)

    # Step 4: Add projects to the solution while respecting resource constraints
    for j in sorted_projects:
        # Create a temporary solution with the current project added
        temp_solution = solution.copy()
        temp_solution[j] = 1
        # Check if the solution remains feasible
        if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
            solution = temp_solution

    # Step 5: Calculate total profit
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit