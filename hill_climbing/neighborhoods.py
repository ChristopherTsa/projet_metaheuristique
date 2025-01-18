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
        np.ndarray: Matrix where each row contains pairs of indices to swap.
    """
    indices = np.arange(N)
    combs = list(combinations(indices, 2 * k))  # Generate all combinations of 2*k items
    swap_indices = []

    for comb in combs:
        # Form pairs from the combinations
        pairs = list(zip(comb[:k], comb[k:2 * k]))
        swap_indices.append(pairs)

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
    # Calculate profit-to-resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included and excluded items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included and excluded items by their ratios
    included_sorted = included_indices[np.argsort(ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    # Limit to k items for replacements
    included_to_consider = included_sorted[:min(k, len(included_sorted))]
    excluded_to_consider = excluded_sorted[:min(k, len(excluded_sorted))]

    # Generate all combinations of replacements
    included_combinations = np.array(list(combinations(included_to_consider, k)))
    excluded_combinations = np.array(list(combinations(excluded_to_consider, k)))

    # Use broadcasting to create all pairs of included and excluded combinations
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Stack results into a single array
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
    # Calculate profit-to-resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included and excluded items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included and excluded items by their ratios
    included_sorted = included_indices[np.argsort(ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(-ratios[excluded_indices])]

    # Limit to k candidates
    worst_included = included_sorted[:k]
    best_excluded = excluded_sorted[:k]

    # Generate all combinations for 1 to k replacements
    included_combinations = np.array(list(chain.from_iterable(combinations(worst_included, m) for m in range(1, k + 1))))
    excluded_combinations = np.array(list(chain.from_iterable(combinations(best_excluded, m) for m in range(1, k + 1))))

    # Use broadcasting to create all possible pairs
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Combine results into a single array
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
    # Calculate profit-to-resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included and excluded items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included and excluded items by their ratios
    included_sorted = included_indices[np.argsort(-ratios[included_indices])]
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    # Limit to k candidates
    best_included = included_sorted[:k]
    worst_excluded = excluded_sorted[:k]

    # Generate all possible replacements using broadcasting
    included_expanded = np.repeat(best_included, len(worst_excluded))
    excluded_expanded = np.tile(worst_excluded, len(best_included))

    # Combine results into a single array
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
    included_combinations = np.array(list(chain.from_iterable(combinations(best_included, m) for m in range(1, k + 1))))
    excluded_combinations = np.array(list(chain.from_iterable(combinations(worst_excluded, m) for m in range(1, k + 1))))

    # Use broadcasting to create all possible pairs
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Combine results into a single array
    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices
