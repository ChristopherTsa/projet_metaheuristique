import numpy as np
from utilities import calculate_profit, is_feasible


def hill_climbing(N, M, initial_solution, resource_consumption, resource_availabilities, profits, generate_neighbors, k):
    """
    Implements the hill climbing algorithm for the multidimensional knapsack problem.
    Optimized with Numba.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        initial_solution (np.ndarray): Initial feasible solution vector (array of 0s and 1s).
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available resources for each resource type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): A function to generate neighbors of the current solution.

    Returns:
        np.ndarray: Final solution vector (array of 0s and 1s).
        float: Total profit of the final solution.
    """
    # Initialize the current solution and profit
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)

    while True:
        neighbors = generate_neighbors(current_solution, profits, resource_consumption, k)
        best_neighbor = current_solution
        best_profit = current_profit

        # Evaluate the neighbors
        for neighbor in neighbors:
            if is_feasible(neighbor, M, resource_consumption, resource_availabilities):
                neighbor_profit = calculate_profit(neighbor, profits)
                if neighbor_profit > best_profit:
                    best_neighbor = neighbor
                    best_profit = neighbor_profit

        # If no better neighbor found, stop
        if np.array_equal(current_solution, best_neighbor):
            break

        # Update the solution with the best neighbor
        current_solution = best_neighbor
        current_profit = best_profit

    return current_solution, current_profit