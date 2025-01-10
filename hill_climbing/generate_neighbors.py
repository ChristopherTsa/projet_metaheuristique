def generate_neighbors(solution, neighborhood_type, k=1):
    """
    Generate neighbors of a solution based on the specified neighborhood type.

    Args:
        solution (list): Current solution (list of 0s and 1s).
        neighborhood_type (str): Type of neighborhood ("flip", "swap", "multi-swap", "hamming").
        k (int): Parameter for neighborhood size (e.g., k-complementation, Hamming distance).

    Returns:
        list: List of neighboring solutions.
    """
    neighbors = []
    N = len(solution)

    if neighborhood_type == "flip":
        for i in range(N):
            neighbor = solution[:]
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)

    elif neighborhood_type == "swap":
        for i in range(N):
            for j in range(i + 1, N):
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)

    elif neighborhood_type == "multi-swap":
        from itertools import combinations
        for indices in combinations(range(N), k):
            neighbor = solution[:]
            for index in indices:
                neighbor[index] = 1 - neighbor[index]
            neighbors.append(neighbor)

    elif neighborhood_type == "hamming":
        from itertools import combinations
        for indices in combinations(range(N), k):
            neighbor = solution[:]
            for index in indices:
                neighbor[index] = 1 - neighbor[index]
            neighbors.append(neighbor)

    else:
        raise ValueError(f"Unsupported neighborhood type: {neighborhood_type}")

    return neighbors
