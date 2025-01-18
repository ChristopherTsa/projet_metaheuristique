import numpy as np
from utilities import calculate_profit


def compute_multipliers(resource_consumption, resource_availabilities):
    """
    Compute resource multipliers (lambda) based on resource tightness.

    Args:
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources (M).

    Returns:
        np.ndarray: Array of lambda multipliers (M), normalized by total tightness.
    """
    # Calculate resource tightness as the ratio of total consumption to availability
    tightness = np.sum(resource_consumption, axis=1) / resource_availabilities

    # Normalize tightness to compute lambda multipliers
    total_tightness = np.sum(tightness)
    multipliers = tightness / total_tightness

    return multipliers


def surrogate_relaxation_mkp(N, resource_consumption, resource_availabilities, profits):
    """
    Solve the surrogate relaxation of the Multiple Knapsack Problem (MKP).

    Args:
        N (int): Number of projects.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources (M).
        profits (np.ndarray): Profits associated with each project.

    Returns:
        np.ndarray: Binary solution array (0 or 1) indicating selected projects.
    """
    # Compute resource multipliers using the optimized function
    multipliers = compute_multipliers(resource_consumption, resource_availabilities)

    # Compute surrogate weights and surrogate capacity using matrix operations
    surrogate_weights = np.dot(multipliers, resource_consumption)  # Weighted sum of resource consumption
    surrogate_capacity = np.dot(multipliers, resource_availabilities)  # Weighted sum of resource availabilities

    # Compute profit-to-weight ratios for valid weights
    valid_weights_mask = surrogate_weights > 0  # Filter out invalid (zero or negative) weights
    ratios = profits[valid_weights_mask] / surrogate_weights[valid_weights_mask]
    valid_indices = np.where(valid_weights_mask)[0]  # Indices of valid weights

    # Sort projects in descending order of profit-to-weight ratio
    sorted_indices = valid_indices[np.argsort(-ratios)]

    # Initialize solution and cumulative weight
    solution = np.zeros(N, dtype=np.int32)
    cumulative_weight = 0

    # Iterate through sorted projects and add to the solution if feasible
    for j in sorted_indices:
        if cumulative_weight + surrogate_weights[j] <= surrogate_capacity:
            solution[j] = 1  # Select the project
            cumulative_weight += surrogate_weights[j]  # Update the cumulative weight

    return solution


def repair_heuristic(initial_solution, resource_consumption, resource_availabilities, profits):
    """
    Repair an infeasible solution for the MKP by satisfying resource constraints.

    Args:
        initial_solution (np.ndarray): Initial binary solution array.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources (M).
        profits (np.ndarray): Profits associated with each project.

    Returns:
        tuple: 
            - np.ndarray: Repaired binary solution.
            - float: Total profit of the repaired solution.
    """
    # Create a copy of the initial solution to avoid modifying the input
    solution = initial_solution.copy()

    # Calculate the total resource usage for each constraint
    total_weights = np.dot(resource_consumption, solution)

    # Repair infeasibility by removing projects until constraints are satisfied
    while np.any(total_weights > resource_availabilities):
        # Identify active projects (selected in the solution)
        active_projects = solution == 1

        # Calculate profit-to-cost ratio for active projects
        resource_costs = np.sum(resource_consumption[:, active_projects], axis=0)
        ratios = profits[active_projects] / resource_costs

        # Identify the project with the smallest profit-to-cost ratio to remove
        item_to_remove = np.where(active_projects)[0][np.argmin(ratios)]

        # Update the solution to remove the selected project
        solution[item_to_remove] = 0

        # Update the total resource usage after removing the project
        total_weights -= resource_consumption[:, item_to_remove]

    # Calculate the total profit of the repaired solution
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit
