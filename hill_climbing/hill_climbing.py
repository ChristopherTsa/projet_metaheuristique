import numpy as np
from numba import njit

@njit
def hill_climbing(N, M, initial_solution, resource_consumption, resource_availabilities, profits, neighborhood_function):
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
        neighborhood_function (function): A function to generate neighbors of the current solution.

    Returns:
        np.ndarray: Final solution vector (array of 0s and 1s).
        float: Total profit of the final solution.
    """
    def calculate_profit(solution):
        """Calculate the total profit of a solution."""
        return np.sum(profits * solution)

    def is_feasible(solution):
        """Check if a solution is feasible."""
        for i in range(M):
            if np.sum(resource_consumption[i, :] * solution) > resource_availabilities[i]:
                return False
        return True

    # Initialize the current solution and profit
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution)

    while True:
        neighbors = neighborhood_function(current_solution)
        best_neighbor = current_solution
        best_profit = current_profit
        
        # Evaluate the neighbors
        for neighbor in neighbors:
            if is_feasible(neighbor):
                neighbor_profit = calculate_profit(neighbor)
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