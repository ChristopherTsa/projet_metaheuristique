def hill_climbing(N, M, initial_solution, resource_consumption, resource_availabilities, profits, neighborhood_type="swap"):
    """
    Implements a hill climbing (local search) algorithm for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        initial_solution (list): Initial feasible solution vector (list of 0s and 1s).
        resource_consumption (list of lists): Resource consumption matrix.
        resource_availabilities (list): List of available quantities for each resource.
        profits (list): List of profits for each project.
        neighborhood_type (str): Type of neighborhood to use ("swap", "add/remove", "multi-swap").

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

    def generate_neighbors(solution, neighborhood_type):
        """Generate neighbors using the specified neighborhood structure."""
        neighbors = []
        if neighborhood_type == "swap":
            for i in range(N):
                neighbor = solution[:]
                neighbor[i] = 1 - neighbor[i]  # Flip inclusion
                neighbors.append(neighbor)
        elif neighborhood_type == "add/remove":
            for i in range(N):
                neighbor = solution[:]
                neighbor[i] = 1 - neighbor[i]  # Add or remove
                neighbors.append(neighbor)
        elif neighborhood_type == "multi-swap":
            for i in range(N):
                for j in range(i + 1, N):
                    neighbor = solution[:]
                    neighbor[i] = 1 - neighbor[i]
                    neighbor[j] = 1 - neighbor[j]
                    neighbors.append(neighbor)
        else:
            raise ValueError("Unsupported neighborhood type.")
        return neighbors

    # Initialize the current solution and profit
    current_solution = initial_solution[:]
    current_profit = calculate_profit(current_solution)

    improved = True

    while improved:
        improved = False
        # Generate neighbors
        neighbors = generate_neighbors(current_solution, neighborhood_type)

        # Explore neighbors to find the best improvement
        for neighbor in neighbors:
            if is_feasible(neighbor):
                neighbor_profit = calculate_profit(neighbor)
                if neighbor_profit > current_profit:
                    current_solution = neighbor
                    current_profit = neighbor_profit
                    improved = True
                    break  # Restart from the new solution

    return current_solution, current_profit

# Example usage:
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

    # Run hill climbing
    final_solution, total_profit = hill_climbing_knapsack(
        N, M, initial_solution, resource_consumption, resource_availabilities, profits, neighborhood_type="swap"
    )

    print("Final Solution:", final_solution)
    print("Total Profit:", total_profit)