import numpy as np
from utilities import calculate_profit, is_feasible


def greedy_heuristic(N, profits, resource_consumption, resource_availabilities):
    
    # Calcul des ratios profit / coût
    total_resources = np.sum(resource_consumption, axis=0)  # Somme des ressources pour chaque projet
    ratios = np.divide(profits, total_resources, where=total_resources > 0, out=np.full(N, np.inf))

    # Tri décroissant des ratios profit / coût
    sorted_projects = np.argsort(ratios)[::-1] 

    # Initialisation de la solution
    solution = np.zeros(N, dtype=np.int32)
    current_resource_usage = np.zeros(resource_consumption.shape[0], dtype=np.float64)

    # Ajout des projets à la solution en respectant les contraintes
    for j in sorted_projects:
        # Vérifier si les ressources restent suffisantes pour ajouter le projet
        additional_usage = current_resource_usage + resource_consumption[:, j]
        if np.all(additional_usage <= resource_availabilities):
            solution[j] = 1
            current_resource_usage = additional_usage

    # Calcul du profit total
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit