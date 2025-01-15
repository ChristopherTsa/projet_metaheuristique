import random
import time
from utilities import calculate_profit, is_feasible
from hill_climbing.hill_climbing import hill_climbing


def vns_hill_climbing(
    N, M, initial_solution, resource_consumption, resource_availabilities, profits, 
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
            # Generate a random neighbor in the k-th neighborhood
            neighbors = generate_neighbors(current_solution, profits, resource_consumption, k)
            
            # Filter neighbors to keep only feasible ones
            feasible_neighbors = [
                neighbor for neighbor in neighbors
                if is_feasible(neighbor, M, resource_consumption, resource_availabilities)
            ]

            # If no feasible neighbors exist, move to the next neighborhood
            if not feasible_neighbors:
                k += 1
                continue

            # Choose a random feasible neighbor
            random_neighbor = random.choice(feasible_neighbors)

            # Apply the Hill Climbing heuristic on the feasible random neighbor
            local_solution, local_profit = hill_climbing(
                N, M, random_neighbor, resource_consumption, resource_availabilities, profits, 
                generate_neighbors, k)
            
            # Update the best solution if local search improves it
            if local_profit > current_profit:
                current_solution = local_solution.copy()
                current_profit = local_profit

                k = 1  # Restart with the first neighborhood
            else:
                k += 1  # Move to the next neighborhood

    return current_solution, current_profit
