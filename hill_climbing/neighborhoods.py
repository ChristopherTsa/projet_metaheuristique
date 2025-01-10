import numpy as np
from itertools import combinations


def multi_swap_neighborhood(solution, k=1):
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


def multi_opt_neighborhood(solution, k=1):
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


def add_remove_neighborhood(solution, k=1):
    """
    Generates the add/remove neighborhood by adding or removing a project.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    for i in range(N):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i]  # Add or remove the project
        neighbors.append(neighbor)
    return neighbors