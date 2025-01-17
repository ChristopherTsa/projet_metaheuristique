import numpy as np
from utilities import calculate_profit, is_feasible


def compute_multipliers(N, M, resource_consumption, resource_availabilities):
    
    
    # Construction de la tension
    tightness = [sum(resource_consumption[i]) / resource_availabilities[i] for i in range(M)]
    
    # Normalisation de la tension pour construire les multiplicateurs lambda
    total_tightness = sum(tightness)
    multipliers = [tight / total_tightness for tight in tightness]
    
    return multipliers

def surrogate_relaxation_mkp(N, M, resource_consumption, resource_availabilities, profits):
    
    
    multipliers = compute_multipliers(N, M, resource_consumption, resource_availabilities)

    # Construction de la contrainte surrogate
    surrogate_weights = [sum(resource_consumption[i][j] * multipliers[i] for i in range(M)) for j in range(N)]
    surrogate_capacity = sum(multipliers[i] * resource_availabilities[i] for i in range(M))
    
    # Constrcution du ratio profit / coût
    ratios = [(profits[j] / surrogate_weights[j], j) for j in range(N) if surrogate_weights[j] > 0]
    ratios.sort(reverse=True, key=lambda x: x[0])  # Tri décroissant des ratios profit / coût
    
    solution = [0] * N
    current_weight = 0
    
    for ratio, j in ratios:
        if current_weight + surrogate_weights[j] <= surrogate_capacity:
            solution[j] = 1
            current_weight += surrogate_weights[j]
    
    return solution


def repair_heuristic(initial_solution, N, M, resource_consumption, resource_availabilities, profits):
    
    solution = initial_solution
    # Construction du poids total pour chaque contrainte
    total_weights = [sum(resource_consumption[i][j] * solution[j] for j in range(N)) for i in range(M)]
    
    # Reparation des contraintes violées
    while any(total_weights[i] > resource_availabilities[i] for i in range(M)):

        # Recherche de projet à enlever en utilisant le plus petit ratio profit / coût 
        ratios = [
            (profits[j] / sum(resource_consumption[i][j] for i in range(M)), j) 
            for j in range(N) if solution[j] == 1
        ]
        ratios.sort(key=lambda x: x[0])  # Tri croissant des ratios profit / coût
        _, item_to_remove = ratios[0]
        
        # Projet retiré
        solution[item_to_remove] = 0
        # Mise à jour des ressources consommées
        total_weights = [sum(resource_consumption[i][j] * solution[j] for j in range(N)) for i in range(M)]

    total_profit = calculate_profit(solution, profits)

    
    return solution, total_profit