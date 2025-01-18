import numpy as np
import time
import random
from utilities import calculate_profit, is_feasible


def simulated_annealing_metaheuristic(N, resource_consumption, resource_availabilities, profits, 
                                  generate_neighbors, initial_solution, max_time, iter_max, initial_temperature=None, cooling_rate=0.95, epsilon=1e-3, k=3):
    """
    Implements the Simulated Annealing metaheuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): Function to generate a neighbor solution.
        max_time (float): Maximum runtime in seconds.
        iter_max (int): Maximum number of iterations before cooling down the temperature.
        initial_temperature (float): Initial temperature for simulated annealing. If None, it is calculated dynamically.
        cooling_rate (float): Rate at which the temperature decreases.
        epsilon (float): Minimum temperature threshold for stopping.
        k (int): Degree of neighborhood.

    Returns:
        np.ndarray: Final solution vector (array of 0s and 1s).
        float: Total profit of the final solution.
    """

    def initialize_temperature(solution):
        """
        Dynamically calculate the initial temperature using feasible neighbors.
        """
        solution_profit = calculate_profit(solution, profits)
        neighbors_indices = generate_neighbors(N, k)
        if neighbors_indices.size == 0:
            return 1000

        deltas = []
        for _ in range(20):  # Limit the number of sampled neighbors
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            neighbor = solution.copy()
            neighbor[random_indices] ^= 1  # Flip bits
            
            # Calculate the profit difference incrementally
            delta_profit = np.sum(profits[random_indices] * (neighbor[random_indices] - solution[random_indices]))
            
            if is_feasible(neighbor, resource_consumption, resource_availabilities) and delta_profit > 0:
                deltas.append(delta_profit)

        mean_delta_f = np.mean(deltas) if deltas else 1
        return -mean_delta_f / np.log(0.8)  # Initial acceptance probability = 0.8


    # Utiliser la solution initiale fournie
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    best_solution = current_solution.copy()
    best_profit = current_profit

    # Initialize temperature
    if initial_temperature is None:
        temperature = initialize_temperature(current_solution)
    else:
        temperature = initial_temperature
    
    start_time = time.time()

    while temperature > epsilon and time.time() - start_time < max_time:
        for _ in range(iter_max):
            # Generate a neighbor solution using the provided function
            neighbors_indices = generate_neighbors(N, k)
            if neighbors_indices.size == 0:  # No neighbors to explore
                current_solution = best_solution.copy()
                current_profit = best_profit
                continue
            
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            neighbor = current_solution.copy()
            neighbor[random_indices] ^= 1  # Flip bits
            
            # Check feasibility
            if not is_feasible(neighbor, resource_consumption, resource_availabilities):
                continue

            # Calculate profit difference
            delta_profit = np.sum(profits[random_indices] * (neighbor[random_indices] - current_solution[random_indices]))
            
            if delta_profit >= 0:
                current_solution[random_indices] = neighbor[random_indices]
                current_profit += delta_profit

                # Update the best solution found
                if current_profit > best_profit:
                    best_solution = current_solution.copy()
                    best_profit = current_profit
                    
            else:
                if np.random.random() < np.exp(delta_profit / temperature):
                    current_solution[random_indices] = neighbor[random_indices]
                    current_profit += delta_profit
        
        # Cool down the temperature
        temperature *= cooling_rate
        
    return best_solution, best_profit