def repair_heuristic(N, M, solution, resource_consumption, resource_availabilities, profits):
    """
    Implements a repair heuristic to adjust an infeasible solution to make it feasible.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        solution (list): Initial solution vector (list of 0s and 1s, may be infeasible).
        resource_consumption (list of lists): Resource consumption matrix.
        resource_availabilities (list): List of available quantities for each resource.
        profits (list): List of profits for each project.

    Returns:
        list: A feasible solution vector (list of 0s and 1s).
        int: Total profit of the repaired solution.
    """
    # Calculate total resource usage for the current solution
    used_resources = [0] * M
    for i in range(M):
        used_resources[i] = sum(resource_consumption[i][j] * solution[j] for j in range(N))

    # Check feasibility
    infeasible = any(used_resources[i] > resource_availabilities[i] for i in range(M))

    # If already feasible, return the solution
    if not infeasible:
        total_profit = sum(profits[j] * solution[j] for j in range(N))
        return solution, total_profit

    # If infeasible, repair the solution
    # Compute profit-to-resource ratios
    ratios = []
    for j in range(N):
        total_resource = sum(resource_consumption[i][j] for i in range(M))
        if total_resource > 0:
            ratios.append(profits[j] / total_resource)
        else:
            ratios.append(0)

    # Sort projects by ascending ratio (least efficient projects first)
    sorted_projects = sorted(range(N), key=lambda x: ratios[x])

    # Remove projects from the solution until it becomes feasible
    for j in sorted_projects:
        if solution[j] == 1:
            # Remove project j
            solution[j] = 0
            for i in range(M):
                used_resources[i] -= resource_consumption[i][j]

            # Check if the solution is now feasible
            if all(used_resources[i] <= resource_availabilities[i] for i in range(M)):
                break

    # Calculate total profit of the repaired solution
    total_profit = sum(profits[j] * solution[j] for j in range(N))

    return solution, total_profit

# Example usage:
# Assuming data is loaded from the previous function:
# data = read_mknap_data("mknap1.txt")
# instance = data[0]  # Use the first instance as an example
# initial_solution = [1] * instance['N']  # Example: all projects initially selected
# repaired_solution, total_profit = repair_heuristic(
#     instance['N'], instance['M'], initial_solution, 
#     instance['resource_consumption'], instance['resource_availabilities'], 
#     instance['profits']
# )
# print("Repaired Solution:", repaired_solution)
# print("Total Profit:", total_profit)
