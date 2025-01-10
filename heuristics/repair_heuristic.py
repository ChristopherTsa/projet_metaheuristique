import numpy as np
from utils import calculate_profit, is_feasible


def surrogate_relaxation_solution(N, M, resource_consumption, resource_availabilities, profits):
    """
    Generate an initial solution using the surrogate relaxation method.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available resources for each resource type.
        profits (np.ndarray): Array of profits for each project.

    Returns:
        np.ndarray: A surrogate relaxation solution (array of 0s and 1s).
    """
    # Compute weights for the surrogate constraint
    weights = resource_availabilities / np.sum(resource_consumption, axis=1)

    # Compute the surrogate profit-to-resource ratio
    surrogate_ratios = np.zeros(N)
    for j in range(N):
        surrogate_resource = np.sum(weights * resource_consumption[:, j])
        if surrogate_resource > 0:
            surrogate_ratios[j] = profits[j] / surrogate_resource
        else:
            surrogate_ratios[j] = 0

    # Sort projects by descending surrogate ratios
    sorted_projects = np.argsort(surrogate_ratios)[::-1]

    # Generate a greedy solution based on surrogate ratios
    solution = np.zeros(N, dtype=np.int32)

    for j in sorted_projects:
        temp_solution = solution.copy()
        temp_solution[j] = 1
        if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
            solution = temp_solution

    return solution


def repair_heuristic(N, M, resource_consumption, resource_availabilities, profits):
    """
    Implements a repair heuristic starting with a surrogate relaxation solution.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available quantities for each resource.
        profits (np.ndarray): Array of profits for each project.

    Returns:
        np.ndarray: A feasible solution vector (array of 0s and 1s).
        float: Total profit of the repaired solution.
    """
    # Generate the initial solution using surrogate relaxation
    solution = surrogate_relaxation_solution(N, M, resource_consumption, resource_availabilities, profits)

    # Check feasibility
    if is_feasible(solution, M, resource_consumption, resource_availabilities):
        total_profit = calculate_profit(solution, profits)
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
            temp_solution = solution.copy()
            temp_solution[j] = 0
            if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
                solution = temp_solution
                break

    # Calculate total profit of the repaired solution
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit