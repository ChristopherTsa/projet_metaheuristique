import numpy as np
from utilities import calculate_profit


def hill_climbing(N, initial_solution, resource_consumption, resource_availabilities, profits, generate_neighbors, k):
    """
    Optimized hill climbing algorithm with minimal storage of modified indices and efficient NumPy operations.

    Args:
        N (int): Number of projects.
        initial_solution (np.ndarray): Initial binary solution array.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources (M).
        profits (np.ndarray): Profits associated with each project.
        generate_neighbors (function): Function to generate neighbor indices for the current solution.
        k (int): Parameter controlling the number of neighbors to generate.

    Returns:
        tuple:
            - np.ndarray: Final solution after optimization.
            - float: Total profit of the optimized solution.
    """
    # Initialize the current solution, associated profit, and resource usage
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    current_resource_usage = np.dot(resource_consumption, current_solution)

    while True:
        # Generate indices of bits to be flipped (neighbors) as a NumPy array
        neighbors_indices = generate_neighbors(N, current_solution, profits, resource_consumption, k)
        if neighbors_indices.size == 0:  # No neighbors to explore
            break

        # Initialize the best profit found during this iteration
        best_profit = current_profit

        # Create a matrix to represent modifications to the solution (+1 or -1)
        delta_matrix = np.zeros((len(neighbors_indices), N), dtype=np.int8)
        row_indices = np.arange(neighbors_indices.shape[0])[:, None]
        delta_matrix[row_indices, neighbors_indices] = 1 - 2 * current_solution[neighbors_indices]

        # Compute incremental changes to resource usage for all neighbors
        delta_resource_usage = resource_consumption @ delta_matrix.T

        # Identify feasible neighbors by checking resource constraints
        feasible_mask = np.all(
            current_resource_usage[:, None] + delta_resource_usage <= resource_availabilities[:, None], 
            axis=0
        )

        if not np.any(feasible_mask):  # No feasible neighbors
            break

        # Extract feasible deltas and compute their associated profit changes
        feasible_deltas = delta_matrix[feasible_mask]
        delta_profits = profits @ feasible_deltas.T
        feasible_profits = current_profit + delta_profits

        # Find the best feasible neighbor
        best_index = np.argmax(feasible_profits)
        best_profit = feasible_profits[best_index]
        best_delta = feasible_deltas[best_index]

        # Terminate if no neighbor improves the current profit
        if best_profit <= current_profit:
            break

        # Update the current solution, profit, and resource usage
        current_solution += best_delta
        current_profit = best_profit
        current_resource_usage += resource_consumption @ best_delta

    return current_solution, current_profit
