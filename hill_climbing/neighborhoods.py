import numpy as np
from itertools import combinations
from utilities import is_feasible


def multi_swap_neighborhood(solution, profits, ressource_consumption, k=1):
    """
    Generates the k-swap neighborhood by swapping k pairs of bits.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).
        k (int): The number of swaps to perform simultaneously.

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    indices = np.arange(N)

    for comb in combinations(indices, 2 * k):  # Generate combinations of 2*k elements
        neighbor = solution.copy()
        pairs = list(zip(comb[:k], comb[k:2 * k]))  # Form pairs of indices to swap
        for i, j in pairs:
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap bits
        neighbors.append(neighbor)

    return neighbors


def multi_opt_neighborhood(solution, profits, ressource_consumption, k=1):
    """
    Generates the k-opt neighborhood by flipping k bits simultaneously.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).
        k (int): The number of bits to flip (k-opt).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []

    for comb in combinations(range(N), k):  # Generate combinations of k elements
        neighbor = solution.copy()
        for idx in comb:
            neighbor[idx] = 1 - neighbor[idx]  # Flip each selected bit
        neighbors.append(neighbor)

    return neighbors


def resource_profit_based_neighborhood(solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing the least efficient items in the solution with the most efficient excluded items.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of items to replace.

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    neighbors = []

    # Calculate profit/resource ratios
    ratios = profits / (np.sum(resource_consumption, axis=0) + 1e-9)

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by ascending ratio
    included_sorted = included_indices[np.argsort(ratios[included_indices])]

    # Sort excluded items by descending ratio
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    # Generate neighbors by replacing k worst included items with k best excluded items
    for i in range(min(k, len(included_sorted))):
        for j in range(min(k, len(excluded_sorted))):
            neighbor = solution.copy()

            # Replace the worst included item with the best excluded item
            neighbor[included_sorted[i]] = 0
            neighbor[excluded_sorted[j]] = 1

            neighbors.append(neighbor)

    return neighbors


from itertools import combinations

def resource_profit_based_k_neighborhood(solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing 1, 2, ..., k pairs of bits (inclusion/exclusion)
    between the worst included items and the best excluded items.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    neighbors = []

    # Calculate profit/resource ratios
    ratios = profits / (np.sum(resource_consumption, axis=0) + 1e-9)

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by ascending ratio
    included_sorted = included_indices[np.argsort(ratios[included_indices])]

    # Sort excluded items by descending ratio
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    # Limit to the top k candidates for both included and excluded
    worst_included = included_sorted[:k]
    best_excluded = excluded_sorted[:k]

    # Generate neighbors by modifying 1, 2, ..., k pairs
    for m in range(1, k + 1):
        included_combinations = list(combinations(worst_included, m))
        excluded_combinations = list(combinations(best_excluded, m))

        for inc_set, exc_set in zip(included_combinations, excluded_combinations):
            neighbor = solution.copy()

            # Apply modifications for m pairs
            for inc_idx, exc_idx in zip(inc_set, exc_set):
                neighbor[inc_idx] = 0  # Exclude the worst included item
                neighbor[exc_idx] = 1  # Include the best excluded item

            neighbors.append(neighbor)

    return neighbors


def resource_profit_based_reverse_neighborhood(solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing the best included items with the worst excluded items.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    neighbors = []

    # Calculate profit/resource ratios
    ratios = profits / (np.sum(resource_consumption, axis=0) + 1e-9)

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by descending ratio (best inclus)
    included_sorted = included_indices[np.argsort(-ratios[included_indices])]

    # Sort excluded items by ascending ratio (worst exclus)
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    # Generate neighbors by replacing k best included items with k worst excluded items
    for i in range(min(k, len(included_sorted))):
        for j in range(min(k, len(excluded_sorted))):
            neighbor = solution.copy()

            # Replace the best included item with the worst excluded item
            neighbor[included_sorted[i]] = 0
            neighbor[excluded_sorted[j]] = 1

            neighbors.append(neighbor)

    return neighbors
