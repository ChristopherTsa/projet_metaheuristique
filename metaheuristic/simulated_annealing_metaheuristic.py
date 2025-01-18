import numpy as np
import time
from utilities import calculate_profit, is_feasible


def simulated_annealing_metaheuristic(
    N, resource_consumption, resource_availabilities, profits, 
    generate_neighbors, initial_solution, max_time, iter_max, 
    initial_temperature=None, cooling_rate=0.95, epsilon=1e-3, k=3
):
    """
    Implements the Simulated Annealing metaheuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): Function to generate a neighbor solution.
        initial_solution (np.ndarray): Initial feasible solution vector (binary array).
        max_time (float): Maximum runtime in seconds.
        iter_max (int): Maximum number of iterations before cooling down the temperature.
        initial_temperature (float, optional): Initial temperature. If None, it is calculated dynamically.
        cooling_rate (float): Rate at which the temperature decreases.
        epsilon (float): Minimum temperature threshold for stopping.
        k (int): Degree of neighborhood.

    Returns:
        tuple:
            - np.ndarray: Final solution vector (binary array of 0s and 1s).
            - float: Total profit of the final solution.
    """

    def initialize_temperature():
        """
        Dynamically calculate the initial temperature using feasible neighbors.

        Returns:
            float: Initial temperature based on average profit differences.
        """
        neighbors_indices = generate_neighbors(N, initial_solution, profits, resource_consumption, k)
        
        if neighbors_indices.size == 0:  # No neighbors available
            return 1000  # Default high temperature

        deltas = []
        for _ in range(20):  # Sample up to 20 random neighbors
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            neighbor = initial_solution.copy()
            neighbor[random_indices] ^= 1  # Flip bits
            
            # Calculate the profit difference incrementally
            delta_profit = np.sum(profits[random_indices] * (neighbor[random_indices] - initial_solution[random_indices]))
            
            if is_feasible(neighbor, resource_consumption, resource_availabilities) and delta_profit > 0:
                deltas.append(delta_profit)

        # Calculate the mean of positive profit differences
        mean_delta_f = np.mean(deltas) if deltas else 1
        return -mean_delta_f / np.log(0.8)  # Set initial acceptance probability to 0.8

    # Initialize the current, best solution, and their respective profits
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    best_solution = current_solution.copy()
    best_profit = current_profit

    # Initialize temperature
    temperature = initialize_temperature() if initial_temperature is None else initial_temperature
    start_time = time.time()  # Record the start time of the algorithm

    # Main loop: continue until the temperature reaches epsilon or max_time is exceeded
    while temperature > epsilon and time.time() - start_time < max_time:
        for _ in range(iter_max):
            # Generate neighbors for the current solution
            neighbors_indices = generate_neighbors(N, current_solution, profits, resource_consumption, k)
            
            if neighbors_indices.size == 0:  # If no neighbors are available, reset to the best solution
                current_solution = best_solution.copy()
                current_profit = best_profit
                continue
            
            # Select a random neighbor
            random_indices = neighbors_indices[np.random.randint(neighbors_indices.shape[0])]
            neighbor = current_solution.copy()
            neighbor[random_indices] ^= 1  # Flip bits
            
            # Ensure the neighbor is feasible
            if not is_feasible(neighbor, resource_consumption, resource_availabilities):
                continue

            # Calculate the profit difference
            delta_profit = np.sum(profits[random_indices] * (neighbor[random_indices] - current_solution[random_indices]))
            
            if delta_profit >= 0:  # Accept if the neighbor improves the profit
                current_solution[random_indices] = neighbor[random_indices]
                current_profit += delta_profit

                # Update the best solution if necessary
                if current_profit > best_profit:
                    best_solution = current_solution.copy()
                    best_profit = current_profit
                    
            else:  # Accept worse solutions with a probability based on the temperature
                if np.random.random() < np.exp(delta_profit / temperature):
                    current_solution[random_indices] = neighbor[random_indices]
                    current_profit += delta_profit
        
        # Cool down the temperature
        temperature *= cooling_rate
        
    return best_solution, best_profit
