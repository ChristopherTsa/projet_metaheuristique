import numpy as np
from utilities import calculate_profit, is_feasible


def greedy_heuristic(N, M, profits, resource_consumption, resource_availabilities):
    
    # Calcul des ratios profit / coût
    ratios = np.zeros(N)
    for j in range(N):
        total_resource = np.sum(resource_consumption[:, j])
        if total_resource > 0:
            ratios[j] = profits[j] / total_resource
        else:
            ratios[j] = np.inf  # les projets sans contraintes de ressources passent en priorité

    # Tri décroissant des ratios profit / coût
    sorted_projects = np.argsort(ratios)[::-1] 

    # Initialisation de la solution
    solution = np.zeros(N, dtype=np.int32)

    # Ajout de projets à la solution en respectant les ressources
    for j in sorted_projects:
        # Création d'une solution temporaire avec le projet courant ajouté
        temp_solution = solution.copy()
        temp_solution[j] = 1
        # Vérifier si la solution est bien faisable
        if is_feasible(temp_solution, M, resource_consumption, resource_availabilities):
            solution = temp_solution

    # Calcul du profit total
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit