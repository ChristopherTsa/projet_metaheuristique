import time
import numpy as np
from utilities import calculate_profit, is_feasible
from hill_climbing.hill_climbing import hill_climbing


def vns_hill_climbing(
    N, initial_solution, resource_consumption, resource_availabilities, profits, 
    generate_neighbors, max_time, k_max
):
    """
    Implements the Variable Neighborhood Search (VNS) algorithm using Hill Climbing as the local search method.

    Args:
        N (int): Number of projects.
        initial_solution (np.ndarray): Initial feasible solution vector (binary array of 0s and 1s).
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each resource type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): Function to generate neighbors based on the current solution.
        max_time (float): Maximum runtime in seconds.
        k_max (int): Maximum degree of neighborhood to explore.

    Returns:
        tuple:
            - np.ndarray: Final solution vector (binary array of 0s and 1s).
            - float: Total profit of the final solution.
    """
    start_time = time.time()  # Record the start time of the algorithm

    # Initialize the current solution and its associated profit
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    
    while time.time() - start_time < max_time:  # Continue until the maximum runtime is reached
        k = 1  # Start with the smallest neighborhood

        while k <= k_max:
            # Generate neighbors for the current neighborhood
            neighbors_indices = generate_neighbors(N, current_solution, profits, resource_consumption, k)

            if neighbors_indices.size == 0:  # If no neighbors are available, move to the next neighborhood
                k += 1
                continue
            
            # Select a random neighbor from the generated neighbors
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            random_neighbor = current_solution.copy()
            random_neighbor[random_indices] ^= 1  # Flip bits using XOR for efficiency

            # Ensure the random neighbor is feasible by checking resource constraints
            while not is_feasible(random_neighbor, resource_consumption, resource_availabilities):
                random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
                random_neighbor = current_solution.copy()
                random_neighbor[random_indices] ^= 1

            # Apply Hill Climbing to improve the feasible random neighbor
            local_solution, local_profit = hill_climbing(
                N, random_neighbor, resource_consumption, resource_availabilities, profits, 
                generate_neighbors, k
            )
            
            # Update the current solution if a better solution is found
            if local_profit > current_profit:
                current_solution = local_solution.copy()
                current_profit = local_profit

                k = 1  # Restart the search with the first neighborhood
            else:
                k += 1  # Move to the next neighborhood

    return current_solution, current_profit
