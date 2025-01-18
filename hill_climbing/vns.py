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
        M (int): Number of resources.
        initial_solution (np.ndarray): Initial feasible solution vector (array of 0s and 1s).
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each resource type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors_list (list of functions): List of neighborhood generation functions.
        max_time (float): Maximum runtime in seconds.
        k_max (int): Maximum degree of neighborhood to explore.

    Returns:
        np.ndarray: Final solution vector (array of 0s and 1s).
        float: Total profit of the final solution.
    """
    start_time = time.time()

    # Initialize the current solution and profit
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    
    while time.time() - start_time < max_time:
        k = 1  # Start with the first neighborhood

        while k <= k_max:
            
            # Générer les indices des bits modifiés sous forme de matrice NumPy
            neighbors_indices = generate_neighbors(N, k)
            if neighbors_indices.size == 0:  # Aucun voisin
                k += 1
                continue
            
            # Générer un voisin aléatoire
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            random_neighbor = current_solution.copy()
            random_neighbor[random_indices] ^= 1 # Inverser les bits directement avec XOR

            # Check if the generated neighbor is feasible
            while not is_feasible(random_neighbor, resource_consumption, resource_availabilities):
                random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
                random_neighbor = current_solution.copy()
                random_neighbor[random_indices] ^= 1 # Inverser les bits directement avec XOR

            # Apply the Hill Climbing heuristic on the feasible random neighbor
            local_solution, local_profit = hill_climbing(
                N, random_neighbor, resource_consumption, resource_availabilities, profits, 
                generate_neighbors, k)
            
            # Update the best solution if local search improves it
            if local_profit > current_profit:
                current_solution = local_solution.copy()
                current_profit = local_profit

                k = 1  # Restart with the first neighborhood
            else:
                k += 1  # Move to the next neighborhood

    return current_solution, current_profit
