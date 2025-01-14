import numpy as np
from utilities import calculate_profit, is_feasible


def surrogate_relaxation_solution(N, M, resource_consumption, resource_availabilities, profits):
    """
    Generate an initial solution using the surrogate relaxation method.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available resources for each resource type.
        profits (np.ndarray): Array of profits for each project.

    Returns:
        np.ndarray: A surrogate relaxation solution (array of 0s and 1s).
    """
    # Compute weights for the surrogate constraint
    weights = resource_availabilities / np.sum(resource_consumption, axis=1)

    # Compute the surrogate profit-to-resource ratio
    surrogate_ratios = np.zeros(N)
    for j in range(N):
        surrogate_resource = np.sum(weights * resource_consumption[:, j])
        if surrogate_resource > 0:
            surrogate_ratios[j] = profits[j] / surrogate_resource
        else:
            surrogate_ratios[j] = 0

    # Sort projects by descending surrogate ratios
    sorted_projects = np.argsort(surrogate_ratios)[::-1]

    # Generate a greedy solution based on surrogate ratios
    solution = np.zeros(N, dtype=np.int32)

    for j in sorted_projects:
        temp_solution = solution.copy()
        temp_solution[j] = 1
        if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
            solution = temp_solution
    return solution

def compute_multipliers(N, M, resource_consumption, resource_availabilities):
    """
    Compute surrogate multipliers based on constraint tightness.
    
    Parameters:
        weights (list of lists): Weight matrix where weights[i][j] is the weight of item j for constraint i.
        capacities (list): Capacities of each constraint.
    
    Returns:
        list: Normalized multipliers for each constraint.
    """
    #m = len(weights)  # Number of constraints
    
    # Compute tightness for each constraint
    tightness = [sum(resource_consumption[i]) / resource_availabilities[i] for i in range(M)]
    
    # Normalize tightness to compute multipliers
    total_tightness = sum(tightness)
    multipliers = [tight / total_tightness for tight in tightness]
    
    return multipliers

def surrogate_relaxation_mkp(N, M, resource_consumption, resource_availabilities, profits):
    """
    Solve the surrogate relaxation of the MKP.
    
    Parameters:
        profits (list): List of item profits.
        weights (list of lists): Weight matrix where weights[i][j] is the weight of item j for constraint i.
        capacities (list): Capacities of each constraint.
        multipliers (list): Surrogate multipliers for each constraint.
        
    Returns:
        list: Initial solution vector (0/1) for the surrogate knapsack problem.
    """
    #n = len(profits)  # Number of items
    #m = len(weights)  # Number of constraints
    multipliers = compute_multipliers(N, M, resource_consumption, resource_availabilities)
    # Compute surrogate weights and capacity
    surrogate_weights = [sum(resource_consumption[i][j] * multipliers[i] for i in range(M)) for j in range(N)]
    surrogate_capacity = sum(multipliers[i] * resource_availabilities[i] for i in range(M))
    
    # Solve surrogate knapsack problem (greedy approach)
    item_indices = list(range(N))
    # Compute profit-to-weight ratio
    ratios = [(profits[j] / surrogate_weights[j], j) for j in range(N) if surrogate_weights[j] > 0]
    ratios.sort(reverse=True, key=lambda x: x[0])  # Sort by descending profit-to-weight ratio
    
    solution = [0] * N
    current_weight = 0
    
    for ratio, j in ratios:
        if current_weight + surrogate_weights[j] <= surrogate_capacity:
            solution[j] = 1
            current_weight += surrogate_weights[j]
    
    return solution


def repair_heuristic_bis(N, M, resource_consumption, resource_availabilities, profits):
    """
    Implements a repair heuristic starting with a surrogate relaxation solution.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Array of available quantities for each resource.
        profits (np.ndarray): Array of profits for each project.

    Returns:
        np.ndarray: A feasible solution vector (array of 0s and 1s).
        float: Total profit of the repaired solution.
    """
    # Generate the initial solution using surrogate relaxation
    #solution = surrogate_relaxation_solution(N, M, resource_consumption, resource_availabilities, profits)
    #solution = np.random.randint(2, size=N)
    solution = surrogate_relaxation_mkp(N, M, resource_consumption, resource_availabilities, profits)
    #solution,  = genetic_metaheuristic(N, M, resource_consumption, resource_availabilities, profits)
    print(solution)
    # Check feasibility
    if is_feasible(solution, M, resource_consumption, resource_availabilities):
        total_profit = calculate_profit(solution, profits)
        return solution, total_profit

    # If infeasible, repair the solution
    # Compute profit-to-resource ratios
    ratios = np.zeros(N)
    for j in range(N):
        total_resource = np.sum(resource_consumption[:, j])
        if total_resource > 0:
            ratios[j] = profits[j] / total_resource
        else:
            ratios[j] = 0

    # Sort projects by ascending ratio (least efficient projects first)
    sorted_projects = np.argsort(ratios)

    # Remove projects from the solution until it becomes feasible
    for j in sorted_projects:
        if solution[j] == 1:
            temp_solution = solution.copy()
            temp_solution[j] = 0
            if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
                solution = temp_solution
                break

    # Calculate total profit of the repaired solution
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit

def repair_heuristic(initial_solution, N, M, resource_consumption, resource_availabilities, profits):
    """
    Repair an infeasible solution for the MKP.
    
    Parameters:
        initial_solution (list): Initial solution vector (0/1).
        weights (list of lists): Weight matrix where weights[i][j] is the weight of item j for constraint i.
        capacities (list): Capacities of each constraint.
        
    Returns:
        list: Feasible solution vector (0/1).
    """
    #n = len(initial_solution)
    #m = len(weights)
    
    #solution = initial_solution[:]
    #solution = surrogate_relaxation_mkp(N, M, resource_consumption, resource_availabilities, profits)
    #solution  = genetic_metaheuristic(N, M, resource_consumption, resource_availabilities, profits)[0]
    solution = initial_solution
    # Compute total weight for each constraint
    total_weights = [sum(resource_consumption[i][j] * solution[j] for j in range(N)) for i in range(M)]
    
    # Repair violations
    while any(total_weights[i] > resource_availabilities[i] for i in range(M)):
        # Find items to remove: lowest profit-to-weight ratio
        ratios = [
            (profits[j] / sum(resource_consumption[i][j] for i in range(M)), j) 
            for j in range(N) if solution[j] == 1
        ]
        ratios.sort(key=lambda x: x[0])  # Sort by ascending profit-to-weight ratio
        _, item_to_remove = ratios[0]
        
        # Remove the item
        solution[item_to_remove] = 0
        # Update weights
        total_weights = [sum(resource_consumption[i][j] * solution[j] for j in range(N)) for i in range(M)]

    total_profit = calculate_profit(solution, profits)

    
    return solution, total_profit