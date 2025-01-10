from numba import njit
import numpy as np


@njit
def one_opt_neighborhood(solution):
    """
    Generates the 1-opt neighborhood for a solution by flipping one bit at a time.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = np.empty((N, N), dtype=np.int32)
    for i in range(N):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i]  # Flip the bit
        neighbors[i] = neighbor
    return neighbors


@njit
def two_opt_neighborhood(solution):
    """
    Generates the 2-opt neighborhood by flipping two bits at a time.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    for i in range(N):
        for j in range(i + 1, N):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]  # Flip the first bit
            neighbor[j] = 1 - neighbor[j]  # Flip the second bit
            neighbors.append(neighbor)
    return neighbors


@njit
def three_opt_neighborhood(solution):
    """
    Generates the 3-opt neighborhood by flipping three bits at a time.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                neighbor = solution.copy()
                neighbor[i] = 1 - neighbor[i]  # Flip the first bit
                neighbor[j] = 1 - neighbor[j]  # Flip the second bit
                neighbor[k] = 1 - neighbor[k]  # Flip the third bit
                neighbors.append(neighbor)
    return neighbors


@njit
def swap_neighborhood(solution):
    """
    Generates the swap neighborhood by swapping two bits.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    for i in range(N):
        for j in range(i + 1, N):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap the bits
            neighbors.append(neighbor)
    return neighbors


@njit
def add_remove_neighborhood(solution):
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


@njit
def add_remove_neighborhood(solution):
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


from itertools import combinations

@njit
def multi_swap_neighborhood(solution, k=2):
    """
    Generates the multi-swap neighborhood by flipping k bits simultaneously.

    Args:
        solution (np.ndarray): The current solution vector (array of 0s and 1s).
        k (int): The number of bits to flip.

    Returns:
        list of np.ndarray: The list of neighbors.
    """
    N = len(solution)
    neighbors = []
    indices = list(range(N))
    for comb in combinations(indices, k):
        neighbor = solution.copy()
        for idx in comb:
            neighbor[idx] = 1 - neighbor[idx]  # Flip each selected bit
        neighbors.append(neighbor)
    return neighbors