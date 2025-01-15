import numpy as np
import time
import random
from utilities import calculate_profit, is_feasible


def simulated_annealing_metaheuristic(N, M, resource_consumption, resource_availabilities, profits, 
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

    def initialize_temperature():
        """
        Dynamically calculate the initial temperature using Method 2 (Kirkpatrick).
        """
        delta_fs = []
        for _ in range(100):  # Test a number of transformations
            solution = np.random.randint(0, 2, N)
            if not is_feasible(solution, M, resource_consumption, resource_availabilities):
                continue
            neighbor = generate_neighbors(solution, profits, resource_consumption, 3)
            if not is_feasible(neighbor, M, resource_consumption, resource_availabilities):
                continue

            delta_f = calculate_profit(neighbor, profits) - calculate_profit(solution, profits)
            if delta_f > 0:
                delta_fs.append(delta_f)

        mean_delta_f = np.mean(delta_fs) if delta_fs else 1
        return -mean_delta_f / np.log(0.8)  # Assuming initial acceptance probability of 0.8

    # Utiliser la solution initiale fournie
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    best_solution = current_solution.copy()
    best_profit = current_profit

    # Initialize temperature
    if initial_temperature is None:
        temperature = initialize_temperature()
    else:
        temperature = initial_temperature
    
    start_time = time.time()

    while temperature > epsilon and time.time() - start_time < max_time:
        
        for i in range(iter_max):
            # Generate a neighbor solution using the provided function
            neighbors = generate_neighbors(current_solution, profits, resource_consumption, k)
            
            # Filter neighbors to keep only feasible ones
            feasible_neighbors = [
                neighbor for neighbor in neighbors
                if is_feasible(neighbor, M, resource_consumption, resource_availabilities)
            ]

            # If no feasible neighbors exist, move to the next neighborhood
            if not feasible_neighbors:
                continue

            # Choose a random feasible neighbor
            neighbor = random.choice(feasible_neighbors)
            neighbor_profit = calculate_profit(neighbor, profits)

            # Decide whether to accept the neighbor
            profit_difference = neighbor_profit - current_profit
            
            if profit_difference >= 0:
                current_solution = neighbor.copy()
                current_profit = neighbor_profit

                # Update the best solution found
                if current_profit > best_profit:
                    best_solution = current_solution.copy()
                    best_profit = current_profit
                    
            else:
                if np.random.random() < np.exp(profit_difference / temperature):
                    current_solution = neighbor.copy()
                    current_profit = neighbor_profit
        
        # Cool down the temperature
        temperature *= cooling_rate
        
    return best_solution, best_profit


def simulated_annealing_random_init_metaheuristic(N, M, resource_consumption, resource_availabilities, profits,
                                  generate_neighbors, initial_temperature=None, cooling_rate=0.95, max_iterations=1000, epsilon=1e-3):
    
    # Generate initial solution
    initial_solution = np.random.randint(0, 2, N)
    while not is_feasible(initial_solution, M, resource_consumption, resource_availabilities):
        initial_solution = np.random.randint(0, 2, N)
    
    best_solution, best_profit = simulated_annealing_metaheuristic(N, M, resource_consumption, resource_availabilities, profits,
                                                                   generate_neighbors, initial_solution, initial_temperature, cooling_rate, max_iterations, epsilon)
    
    return best_solution, best_profit
