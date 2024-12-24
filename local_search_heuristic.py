def local_search_heuristic(N, M, solution, resource_consumption, resource_availabilities, profits):
    """
    Implements a local search heuristic for the multidimensional knapsack problem.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        solution (list): Initial feasible solution vector (list of 0s and 1s).
        resource_consumption (list of lists): Resource consumption matrix.
        resource_availabilities (list): List of available quantities for each resource.
        profits (list): List of profits for each project.

    Returns:
        list: An improved feasible solution vector (list of 0s and 1s).
        int: Total profit of the improved solution.
    """
    def is_feasible(solution):
        """Check if a solution is feasible."""
        for i in range(M):
            if sum(resource_consumption[i][j] * solution[j] for j in range(N)) > resource_availabilities[i]:
                return False
        return True

    def calculate_profit(solution):
        """Calculate the total profit of a solution."""
        return sum(profits[j] * solution[j] for j in range(N))

    # Start with the initial solution
    current_solution = solution[:]
    current_profit = calculate_profit(current_solution)

    improved = True

    while improved:
        improved = False
        for j in range(N):
            # Flip the selection status of project j
            neighbor_solution = current_solution[:]
            neighbor_solution[j] = 1 - neighbor_solution[j]

            # Check if the neighbor solution is feasible
            if is_feasible(neighbor_solution):
                neighbor_profit = calculate_profit(neighbor_solution)

                # If the neighbor solution has better profit, adopt it
                if neighbor_profit > current_profit:
                    current_solution = neighbor_solution
                    current_profit = neighbor_profit
                    improved = True
                    break  # Exit the loop to restart from the new solution

    return current_solution, current_profit

# Example usage:
# Assuming data is loaded from the previous function:
# data = read_mknap_data("mknap1.txt")
# instance = data[0]  # Use the first instance as an example
# initial_solution = [0] * instance['N']  # Example: Start with no projects selected
# repaired_solution, _ = repair_heuristic(
#     instance['N'], instance['M'], initial_solution, 
#     instance['resource_consumption'], instance['resource_availabilities'], 
#     instance['profits']
# )
# improved_solution, total_profit = local_search_knapsack(
#     instance['N'], instance['M'], repaired_solution, 
#     instance['resource_consumption'], instance['resource_availabilities'], 
#     instance['profits']
# )
# print("Improved Solution:", improved_solution)
# print("Total Profit:", total_profit)
