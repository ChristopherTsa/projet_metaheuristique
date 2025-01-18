import numpy as np
from itertools import chain, combinations


def multi_swap_neighborhood_indices(N, solution, profits, resource_consumption, k=1):
    """
    Génère les paires d'indices à échanger pour un voisinage k-swap sous forme de tableau NumPy.

    Args:
        N (int): Taille de la solution.
        k (int): Nombre de swaps à effectuer simultanément.

    Returns:
        np.ndarray: Matrice où chaque ligne contient les indices à échanger sous forme de paires.
    """
    indices = np.arange(N)
    combs = list(combinations(indices, 2 * k))  # Générer toutes les combinaisons de 2*k éléments
    swap_indices = []

    for comb in combs:
        pairs = list(zip(comb[:k], comb[k:2 * k]))  # Former les paires à partir des combinaisons
        swap_indices.append(pairs)

    # Convertir en tableau NumPy avec une structure standardisée
    return np.array(swap_indices, dtype=np.int32)


def multi_opt_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Génère les indices des bits modifiés sous forme de tableau NumPy.

    Args:
        solution (np.ndarray): Solution courante.
        k (int): Nombre de bits à inverser.

    Returns:
        np.ndarray: Matrice où chaque ligne contient les indices modifiés.
    """
    combs = list(combinations(range(N), k))  # Générer les combinaisons
    return np.array(combs, dtype=np.int32)  # Convertir en tableau NumPy


def resource_profit_based_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing the least efficient items in the solution 
    with the most efficient excluded items.
    Returns an array where each element is an array of indices to change.

    Args:
        N (int): Total number of items.
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of items to replace.

    Returns:
        np.ndarray: A 2D array where each row contains the indices to change.
    """
    # Calculate profit/resource ratios using np.divide
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by ascending ratio
    included_sorted = included_indices[np.argsort(ratios[included_indices])]

    # Sort excluded items by descending ratio
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
    Generates a neighborhood by replacing 1, 2, ..., k pairs of bits (inclusion/exclusion)
    between the worst included items and the best excluded items, without using an explicit loop.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices of items to exclude and include.
    """
    # Calculate profit/resource ratios using np.divide
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

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

    # Generate all combinations for 1 to k
    included_combinations = np.array(list(chain.from_iterable(combinations(worst_included, m) for m in range(1, k + 1))))
    excluded_combinations = np.array(list(chain.from_iterable(combinations(best_excluded, m) for m in range(1, k + 1))))

    # Use broadcasting to create all possible pairs
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Stack results into a single array
    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices


def resource_profit_based_reverse_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing the best included items with the worst excluded items.
    Returns an array where each element contains the indices of items to exclude and include.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices to exclude and include.
    """
    # Calculate profit/resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by descending ratio
    included_sorted = included_indices[np.argsort(-ratios[included_indices])]

    # Sort excluded items by ascending ratio
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    # Limit to k candidates for replacements
    best_included = included_sorted[:k]
    worst_excluded = excluded_sorted[:k]

    # Generate all possible neighbors using broadcasting
    included_expanded = np.repeat(best_included, len(worst_excluded))
    excluded_expanded = np.tile(worst_excluded, len(best_included))

    # Combine results into a single array
    neighborhood_indices = np.stack((included_expanded, excluded_expanded), axis=1)

    return neighborhood_indices


def resource_profit_based_reverse_k_neighborhood(N, solution, profits, resource_consumption, k=1):
    """
    Generates a neighborhood by replacing 1, 2, ..., k pairs of bits (inclusion/exclusion)
    between the best included items and the worst excluded items.

    Args:
        solution (np.ndarray): Current solution vector (array of 0s and 1s).
        profits (np.ndarray): Array of profits for each item.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        k (int): Number of candidates to consider for replacements.

    Returns:
        np.ndarray: A 2D array where each row contains indices of items to exclude and include.
    """
    # Calculate profit/resource ratios
    total_resources = np.sum(resource_consumption, axis=0)
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(profits.shape, np.inf))

    # Identify indices of included (x_i = 1) and excluded (x_i = 0) items
    included_indices = np.where(solution == 1)[0]
    excluded_indices = np.where(solution == 0)[0]

    # Sort included items by descending ratio
    included_sorted = included_indices[np.argsort(-ratios[included_indices])]

    # Sort excluded items by ascending ratio
    excluded_sorted = excluded_indices[np.argsort(ratios[excluded_indices])]

    # Limit to the top k candidates for replacements
    best_included = included_sorted[:k]
    worst_excluded = excluded_sorted[:k]

    # Generate all combinations for 1, 2, ..., k pairs
    included_combinations = np.array(list(chain.from_iterable(combinations(best_included, m) for m in range(1, k + 1))))
    excluded_combinations = np.array(list(chain.from_iterable(combinations(worst_excluded, m) for m in range(1, k + 1))))

    # Use broadcasting to create all possible pairs
    included_expanded = np.repeat(included_combinations, len(excluded_combinations), axis=0)
    excluded_expanded = np.tile(excluded_combinations, (len(included_combinations), 1))

    # Combine results into a single array
    neighborhood_indices = np.hstack((included_expanded, excluded_expanded))

    return neighborhood_indices