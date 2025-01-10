import numpy as np
from numba import njit

@njit
def simulated_annealing_metaheuristic(N, M, initial_solution, resource_consumption, resource_availabilities, profits, 
                                  initial_temperature=1000, cooling_rate=0.95, max_iterations=1000):
    """
    Implements the Simulated Annealing metaheuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        initial_solution (np.ndarray): Initial feasible solution vector (array of 0s and 1s).
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each project.
        initial_temperature (float): Initial temperature for simulated annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        max_iterations (int): Maximum number of iterations for the algorithm.

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

    def generate_neighbor(solution):
        """Generate a random neighbor by flipping a single bit."""
        neighbor = solution.copy()
        j = np.random.randint(0, N)
        neighbor[j] = 1 - neighbor[j]  # Flip inclusion status
        return neighbor

    # Initialize the current solution and profit
    current_solution = initial_solution.copy()
    if not is_feasible(current_solution):
        raise ValueError("Initial solution must be feasible.")
    
    current_profit = calculate_profit(current_solution)
    best_solution = current_solution.copy()
    best_profit = current_profit

    temperature = initial_temperature

    for _ in range(max_iterations):
        # Generate a neighbor solution
        neighbor = generate_neighbor(current_solution)
        if is_feasible(neighbor):
            neighbor_profit = calculate_profit(neighbor)

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
        if temperature < 1e-3:
            break

    return best_solution, best_profit