import numpy as np
from numba import njit


@njit
def repair_heuristic(N, M, solution, resource_consumption, resource_availabilities, profits):
    """
    Implements a repair heuristic to adjust an infeasible solution to make it feasible.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        solution (np.ndarray): Initial solution vector (array of 0s and 1s, may be infeasible).
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available quantities for each resource.
        profits (np.ndarray): Array of profits for each project.

    Returns:
        np.ndarray: A feasible solution vector (array of 0s and 1s).
        float: Total profit of the repaired solution.
    """
    # Calculate total resource usage for the current solution
    used_resources = np.zeros(M)
    for i in range(M):
        used_resources[i] = np.sum(resource_consumption[i, :] * solution)

    # Check feasibility
    infeasible = np.any(used_resources > resource_availabilities)

    # If already feasible, return the solution
    if not infeasible:
        total_profit = np.sum(profits * solution)
        return solution, total_profit

    # If infeasible, repair the solution
    # Compute profit-to-resource ratios
    ratios = np.zeros(N)
    for j in range(N):
        total_resource = np.sum(resource_consumption[:, j])
        if total_resource > 0:
            ratios[j] = profits[j] / total_resource
        else:
            ratios[j] = 0

    # Sort projects by ascending ratio (least efficient projects first)
    sorted_projects = np.argsort(ratios)

    # Remove projects from the solution until it becomes feasible
    for j in sorted_projects:
        if solution[j] == 1:
            # Remove project j
            solution[j] = 0
            for i in range(M):
                used_resources[i] -= resource_consumption[i, j]

            # Check if the solution is now feasible
            if np.all(used_resources <= resource_availabilities):
                break

    # Calculate total profit of the repaired solution
    total_profit = np.sum(profits * solution)

    return solution, total_profit


# Example usage:
# Assuming data is loaded from the previous function:
# data = read_knapsack_data("mknap1.txt")
# instance = data[0]  # Use the first instance as an example
# initial_solution = [1] * instance['N']  # Example: all projects initially selected
# repaired_solution, total_profit = repair_heuristic(
#     instance['N'], instance['M'], initial_solution, 
#     instance['resource_consumption'], instance['resource_availabilities'], 
#     instance['profits']
# )
# print("Repaired Solution:", repaired_solution)
# print("Total Profit:", total_profit)
