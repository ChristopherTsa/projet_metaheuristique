import numpy as np
from utilities import calculate_profit, is_feasible


def simulated_annealing_metaheuristic(N, M, resource_consumption, resource_availabilities, profits, 
                                  generate_neighbors, initial_solution, initial_temperature=None, cooling_rate=0.95, max_iterations=1000, epsilon=1e-3):
    """
    Implements the Simulated Annealing metaheuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): Function to generate a neighbor solution.
        initial_temperature (float): Initial temperature for simulated annealing. If None, it is calculated dynamically.
        cooling_rate (float): Rate at which the temperature decreases.
        max_iterations (int): Maximum number of iterations for the algorithm.
        epsilon (float): Minimum temperature threshold for stopping.

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
            neighbor = generate_neighbors(solution, 3)
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

    iteration = 0

    while temperature > epsilon and iteration < max_iterations:
        # Generate a neighbor solution using the provided function
        neighbor = generate_neighbors(current_solution, 3)
        if is_feasible(neighbor, M, resource_consumption, resource_availabilities):
            neighbor_profit = calculate_profit(neighbor, profits)

            # Decide whether to accept the neighbor
            profit_difference = neighbor_profit - current_profit
            if profit_difference > 0 or np.random.random() < np.exp(profit_difference / temperature):
                current_solution = neighbor
                current_profit = neighbor_profit

                # Update the best solution found
                if current_profit > best_profit:
                    best_solution = current_solution.copy()
                    best_profit = current_profit

        # Cool down the temperature
        temperature *= cooling_rate
        iteration += 1

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
