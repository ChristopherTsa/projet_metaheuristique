import numpy as np
from numba import njit

@njit
def nearest_neighbor_heuristic(N, M, profits, resource_consumption, resource_availabilities):
    """
    Implements the nearest neighbor heuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        profits (np.ndarray): Array of profits for each project.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.

    Returns:
        np.ndarray: A feasible solution vector (array of 0s and 1s).
        float: Total profit of the solution.
    """
    # Initialize solution and resource usage
    solution = np.zeros(N, dtype=np.int32)
    used_resources = np.zeros(M, dtype=np.float64)
    total_profit = 0.0

    # Calculate profit-to-resource ratios
    ratios = np.zeros(N, dtype=np.float64)
    for j in range(N):
        total_resource = np.sum(resource_consumption[:, j])
        if total_resource > 0:
            ratios[j] = profits[j] / total_resource
        else:
            ratios[j] = 0

    # Sort projects by descending ratio of profit-to-resource
    sorted_indices = np.argsort(ratios)[::-1]

    # Add projects to the solution if they fit within the resource constraints
    for j in sorted_indices:
        fits = True
        for i in range(M):
            if used_resources[i] + resource_consumption[i, j] > resource_availabilities[i]:
                fits = False
                break
        if fits:
            solution[j] = 1
            for i in range(M):
                used_resources[i] += resource_consumption[i, j]
            total_profit += profits[j]

    return solution, total_profit