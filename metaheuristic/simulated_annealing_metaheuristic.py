import random
import math

def simulated_annealing_metaheuristic(N, M, initial_solution, resource_consumption, resource_availabilities, profits, 
                                  initial_temperature=1000, cooling_rate=0.95, max_iterations=1000):
    """
    Implements the Simulated Annealing metaheuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        initial_solution (list): Initial feasible solution vector (list of 0s and 1s).
        resource_consumption (list of lists): Resource consumption matrix.
        resource_availabilities (list): List of available quantities for each resource.
        profits (list): List of profits for each project.
        initial_temperature (float): Initial temperature for simulated annealing.
        cooling_rate (float): Rate at which the temperature decreases.
        max_iterations (int): Maximum number of iterations for the algorithm.

    Returns:
        list: Final solution vector.
        int: Total profit of the final solution.
    """
    def calculate_profit(solution):
        """Calculate the total profit of a solution."""
        return sum(profits[j] * solution[j] for j in range(N))

    def is_feasible(solution):
        """Check if a solution is feasible."""
        for i in range(M):
            if sum(resource_consumption[i][j] * solution[j] for j in range(N)) > resource_availabilities[i]:
                return False
        return True

    def generate_neighbor(solution):
        """Generate a random neighbor by flipping a single bit."""
        neighbor = solution[:]
        j = random.randint(0, N - 1)
        neighbor[j] = 1 - neighbor[j]  # Flip inclusion status
        return neighbor

    # Initialize the current solution and profit
    current_solution = initial_solution[:]
    if not is_feasible(current_solution):
        raise ValueError("Initial solution must be feasible.")
    
    current_profit = calculate_profit(current_solution)
    best_solution = current_solution[:]
    best_profit = current_profit

    temperature = initial_temperature

    for _ in range(max_iterations):
        # Generate a neighbor solution
        neighbor = generate_neighbor(current_solution)
        if is_feasible(neighbor):
            neighbor_profit = calculate_profit(neighbor)

            # Decide whether to accept the neighbor
            profit_difference = neighbor_profit - current_profit
            if profit_difference > 0 or random.random() < math.exp(profit_difference / temperature):
                current_solution = neighbor
                current_profit = neighbor_profit

                # Update the best solution found
                if current_profit > best_profit:
                    best_solution = current_solution[:]
                    best_profit = current_profit

        # Cool down the temperature
        temperature *= cooling_rate
        if temperature < 1e-3:
            break

    return best_solution, best_profit

# Example usage
if __name__ == "__main__":
    # Example data for testing
    N = 5  # Number of projects
    M = 3  # Number of resources
    profits = [20, 30, 50, 10, 40]
    resource_consumption = [
        [10, 20, 30, 10, 40],  # Resource 1 consumption
        [20, 10, 40, 10, 30],  # Resource 2 consumption
        [30, 20, 10, 40, 20],  # Resource 3 consumption
    ]
    resource_availabilities = [100, 120, 150]  # Resource constraints
    initial_solution = [1, 0, 1, 0, 0]  # Example feasible solution

    # Run simulated annealing
    final_solution, total_profit = simulated_annealing_knapsack(
        N, M, initial_solution, resource_consumption, resource_availabilities, profits
    )

    print("Final Solution:", final_solution)
    print("Total Profit:", total_profit)