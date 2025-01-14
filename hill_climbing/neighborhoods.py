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


def resource_profit_based_1_pair_neighborhood(solution, profits, resource_consumption, k=1):
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


def resource_profit_based_k_pair_neighborhood(solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing k least efficient items in the solution with k most efficient excluded items.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of pairs to replace.

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

    # Generate all possible combinations of k pairs
    included_combinations = list(combinations(included_sorted, k))
    excluded_combinations = list(combinations(excluded_sorted, k))

    # Generate neighbors by replacing k worst included items with k best excluded items
    for included_set in included_combinations:
        for excluded_set in excluded_combinations:
            neighbor = solution.copy()

            # Replace the selected included items with the selected excluded items
            for included_idx, excluded_idx in zip(included_set, excluded_set):
                neighbor[included_idx] = 0
                neighbor[excluded_idx] = 1

            neighbors.append(neighbor)

    return neighbors