import numpy as np
from utilities import calculate_profit, is_feasible


def hill_climbing(N, M, initial_solution, resource_consumption, resource_availabilities, profits, generate_neighbors, k):
    
    # Initialisation de la solution  courante et du profit associé
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, profits)

    while True:
        neighbors = generate_neighbors(current_solution, profits, resource_consumption, k)
        best_neighbor = current_solution
        best_profit = current_profit

        # Evaluation des voisins
        for neighbor in neighbors:
            if is_feasible(neighbor, M, resource_consumption, resource_availabilities):
                neighbor_profit = calculate_profit(neighbor, profits)
                if neighbor_profit > best_profit:
                    best_neighbor = neighbor
                    best_profit = neighbor_profit

        # Si il n'y a pas de meilleur voisin, stop
        if np.array_equal(current_solution, best_neighbor):
            break

        # Mise à jour de la solution avec le meilleur voisin
        current_solution = best_neighbor
        current_profit = best_profit

    return current_solution, current_profit