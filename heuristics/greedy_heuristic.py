import numpy as np
from utilities import calculate_profit


def greedy_heuristic(N, profits, resource_consumption, resource_availabilities):
    """
    Implements a greedy heuristic for selecting projects based on their profit-to-cost ratio.

    Parameters:
    N (int): Number of projects.
    profits (array-like): Array of profits associated with each project.
    resource_consumption (2D array): Matrix where each column represents the resource consumption of a project.
    resource_availabilities (array-like): Array of available resources for the projects.

    Returns:
    tuple:
        - solution (array): Binary array indicating selected projects (1 if selected, 0 otherwise).
        - total_profit (float): Total profit obtained from the selected projects.
    """
    # Calculate the profit-to-cost ratio for each project
    total_resources = np.sum(resource_consumption, axis=0)  # Sum of resource usage for each project
    ratios = np.divide(
        profits,
        total_resources,
        where=total_resources > 0,  # Avoid division by zero
        out=np.full(N, np.inf)  # Assign infinite ratio to projects with zero total cost
    )

    # Sort projects in descending order of their profit-to-cost ratio
    sorted_projects = np.argsort(ratios)[::-1]

    # Initialize the solution array and track current resource usage
    solution = np.zeros(N, dtype=np.int32)  # Binary array to indicate selected projects
    current_resource_usage = np.zeros(resource_consumption.shape[0], dtype=np.float64)  # Current usage per resource

    # Iterate over sorted projects and add them to the solution if feasible
    for j in sorted_projects:
        # Calculate the resource usage if this project is added
        additional_usage = current_resource_usage + resource_consumption[:, j]

        # Check if adding this project is within the available resources
        if np.all(additional_usage <= resource_availabilities):
            solution[j] = 1  # Mark project as selected
            current_resource_usage = additional_usage  # Update current resource usage

    # Compute the total profit of the selected projects
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit
