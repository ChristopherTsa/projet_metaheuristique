import numpy as np
from itertools import chain, combinations


def multi_swap_neighborhood_indices(N, solution, profits, resource_consumption, k=1):
    """
    Generate pairs of indices to swap for a k-swap neighborhood as a NumPy array.

    Args:
        N (int): Size of the solution.
        solution (np.ndarray): Current solution vector.
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix.
        k (int): Number of swaps to perform simultaneously.

    Returns:
        np.ndarray: Array of shape (num_combinations, 2*k) containing indices to swap.
    """
    indices = np.arange(N)
    combs = list(combinations(indices, 2 * k))  # Generate all combinations of 2*k items
    swap_indices = []

    for comb in combs:
        # Flatten the swap pairs into a single array
        swap_indices.append(comb)

    # Convert to a standardized NumPy array
    return np.array(swap_indices, dtype=np.int32)


def multi_opt_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generate indices of modified bits as a NumPy array for a k-opt neighborhood.

    Args:
        N (int): Size of the solution.
        solution (np.ndarray): Current solution vector.
        k (int): Number of bits to flip.

    Returns:
        np.ndarray: Matrix where each row contains indices of bits to flip.
    """
    combs = list(combinations(range(N), k))  # Generate all combinations of k items
    return np.array(combs, dtype=np.int32)


def resource_profit_based_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generate a neighborhood by replacing the least efficient items in the solution
    with the most efficient excluded items.

    Args:
        N (int): Total number of items.
        solution (np.ndarray): Current solution vector (binary array).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of items to replace.

    Returns:
        np.ndarray: A 2D array where each row contains indices to change.
    """
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    included_sorted = included_indices[np.argsort(ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    # Limit to k items
    included_to_consider = included_sorted[:k]
    excluded_to_consider = excluded_sorted[:k]

    if len(included_to_consider) == 0 or len(excluded_to_consider) == 0:
        return np.empty((0, k * 2), dtype=np.int32)

    # Generate all combinations of replacements
    included_combinations = np.array(list(combinations(included_to_consider, k)))
    excluded_combinations = np.array(list(combinations(excluded_to_consider, k)))

    # Pad combinations to ensure consistent dimensions
    max_len = max(len(included_combinations), len(excluded_combinations))
    included_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in included_combinations])
    excluded_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in excluded_combinations])

    # Combine replacements
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices


def resource_profit_based_k_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generate a neighborhood by replacing 1, 2, ..., k pairs of bits (inclusion/exclusion)
    between the worst included items and the best excluded items.

    Args:
        N (int): Total number of items.
        solution (np.ndarray): Current solution vector (binary array).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices to exclude and include.
    """
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    included_sorted = included_indices[np.argsort(ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    worst_included = included_sorted[:k]
    best_excluded = excluded_sorted[:k]

    included_combinations = list(chain.from_iterable(combinations(worst_included, m) for m in range(1, k + 1)))
    excluded_combinations = list(chain.from_iterable(combinations(best_excluded, m) for m in range(1, k + 1)))

    if not included_combinations or not excluded_combinations:
        return np.empty((0, 2 * k), dtype=np.int32)

    included_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in included_combinations])
    excluded_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in excluded_combinations])

    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices


def resource_profit_based_reverse_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generate a neighborhood by replacing the best included items with the worst excluded items.

    Args:
        N (int): Total number of items.
        solution (np.ndarray): Current solution vector (binary array).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices to exclude and include.
    """
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    included_sorted = included_indices[np.argsort(-ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    best_included = included_sorted[:k]
    worst_excluded = excluded_sorted[:k]

    if len(best_included) == 0 or len(worst_excluded) == 0:
        return np.empty((0, 2 * k), dtype=np.int32)

    included_expanded = np.repeat(best_included, len(worst_excluded))
    excluded_expanded = np.tile(worst_excluded, len(best_included))

    neighborhood_indices = np.stack((included_expanded, excluded_expanded), axis=1)

    return neighborhood_indices


def resource_profit_based_reverse_k_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generate a neighborhood by replacing 1, 2, ..., k pairs of bits (inclusion/exclusion)
    between the best included items and the worst excluded items.

    Args:
        N (int): Total number of items.
        solution (np.ndarray): Current solution vector (binary array).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices to exclude and include.
    """
    # Calculate profit-to-resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included and excluded items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included and excluded items by their ratios
    included_sorted = included_indices[np.argsort(-ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    # Limit to top k candidates
    best_included = included_sorted[:k]
    worst_excluded = excluded_sorted[:k]

    # Generate combinations for 1, 2, ..., k pairs
    included_combinations = list(chain.from_iterable(combinations(best_included, m) for m in range(1, k + 1)))
    excluded_combinations = list(chain.from_iterable(combinations(worst_excluded, m) for m in range(1, k + 1)))

    # Handle cases where combinations are empty
    if not included_combinations or not excluded_combinations:
        return np.empty((0, 2 * k), dtype=np.int32)

    # Pad combinations to ensure consistent dimensions
    max_len = max(len(included_combinations), len(excluded_combinations))
    included_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in included_combinations])
    excluded_combinations = np.array([list(comb) + [0] * (k - len(comb)) for comb in excluded_combinations])

    # Use broadcasting to create all possible pairs
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Combine results into a single array
    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices
