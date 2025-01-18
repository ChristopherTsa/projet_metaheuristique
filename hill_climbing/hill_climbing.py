import numpy as np
from utilities import calculate_profit


def hill_climbing(N, initial_solution, resource_consumption, resource_availabilities, profits, generate_neighbors, k):
    """
    Hill climbing optimisé avec stockage minimal des indices modifiés et utilisation de NumPy.
    """
    # Initialisation de la solution courante et du profit associé
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)
    current_resource_usage = np.dot(resource_consumption, current_solution)

    while True:
        # Générer les indices des bits modifiés sous forme de matrice NumPy
        neighbors_indices = generate_neighbors(N, current_solution, profits, resource_consumption, k)
        if neighbors_indices.size == 0:  # Aucun voisin
            break

        # Initialisation des meilleures solutions
        best_profit = current_profit

        # Matrice des modifications (+1 ou -1)
        delta_matrix = np.zeros((len(neighbors_indices), N), dtype=np.int8)
        row_indices = np.arange(neighbors_indices.shape[0])[:, None]
        delta_matrix[row_indices, neighbors_indices] = 1 - 2 * current_solution[neighbors_indices]

        # Calcul incrémental vectorisé
        delta_resource_usage = resource_consumption @ delta_matrix.T
        feasible_mask = np.all(current_resource_usage[:, None] + delta_resource_usage <= resource_availabilities[:, None], axis=0)

        if not np.any(feasible_mask):  # Aucun voisin faisable
            break

        feasible_deltas = delta_matrix[feasible_mask]
        delta_profits = profits @ feasible_deltas.T
        feasible_profits = current_profit + delta_profits

        # Trouver le meilleur voisin faisable
        best_index = np.argmax(feasible_profits)
        best_profit = feasible_profits[best_index]
        best_delta = feasible_deltas[best_index]

        # Si aucun voisin n'améliore le profit, stop
        if best_profit <= current_profit:
            break

        # Mise à jour de la solution courante
        current_solution += best_delta
        current_profit = best_profit
        current_resource_usage += resource_consumption @ best_delta

    return current_solution, current_profit