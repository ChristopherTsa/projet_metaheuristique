import numpy as np
from utilities import calculate_profit


def compute_multipliers(resource_consumption, resource_availabilities):
    """
    Calcule les multiplicateurs lambda basés sur la tension des ressources.

    Args:
        N (int): Nombre de projets.
        M (int): Nombre de ressources.
        resource_consumption (np.ndarray): Consommation des ressources (M x N).
        resource_availabilities (np.ndarray): Disponibilités des ressources (M).

    Returns:
        np.ndarray: Tableau des multiplicateurs lambda (M).
    """
    # Calcul vectorisé de la tension
    tightness = np.sum(resource_consumption, axis=1) / resource_availabilities

    # Normalisation pour obtenir les multiplicateurs lambda
    total_tightness = np.sum(tightness)
    multipliers = tightness / total_tightness

    return multipliers


def surrogate_relaxation_mkp(N, resource_consumption, resource_availabilities, profits):
    """
    Résolution de la relaxation surrogate du problème MKP.

    Args:
        N (int): Nombre de projets.
        M (int): Nombre de ressources.
        resource_consumption (np.ndarray): Matrice de consommation des ressources (M x N).
        resource_availabilities (np.ndarray): Disponibilité des ressources (M).
        profits (np.ndarray): Profits associés aux projets.

    Returns:
        np.ndarray: Solution binaire (0 ou 1) pour chaque projet.
    """
    # Calcul des multiplicateurs avec la fonction optimisée
    multipliers = compute_multipliers(resource_consumption, resource_availabilities)

    # Calcul vectorisé des poids surrogates et de la capacité surrogate
    surrogate_weights = np.dot(multipliers, resource_consumption)  # Produit matriciel pour les poids
    surrogate_capacity = np.dot(multipliers, resource_availabilities)  # Produit scalaire pour la capacité

    # Calcul vectorisé des ratios profit / poids
    valid_weights_mask = surrogate_weights > 0  # Filtrer les poids valides
    ratios = profits[valid_weights_mask] / surrogate_weights[valid_weights_mask]
    valid_indices = np.where(valid_weights_mask)[0]  # Indices des poids valides

    # Tri des projets par ordre décroissant de ratios
    sorted_indices = valid_indices[np.argsort(-ratios)]  # Indices triés selon les ratios décroissants

    # Construction de la solution
    solution = np.zeros(N, dtype=np.int32)
    cumulative_weight = 0

    # Parcours vectorisé des projets triés pour respecter la capacité
    for j in sorted_indices:
        if cumulative_weight + surrogate_weights[j] <= surrogate_capacity:
            solution[j] = 1
            cumulative_weight += surrogate_weights[j]

    return solution


def repair_heuristic(initial_solution, resource_consumption, resource_availabilities, profits):
    """
    Répare une solution pour le problème MKP en respectant les contraintes.

    Args:
        initial_solution (np.ndarray): Solution initiale (binaire).
        N (int): Nombre de projets.
        M (int): Nombre de ressources.
        resource_consumption (np.ndarray): Consommation des ressources (M x N).
        resource_availabilities (np.ndarray): Disponibilités des ressources (M).
        profits (np.ndarray): Profits associés aux projets.

    Returns:
        tuple: Solution réparée (binaire) et profit total.
    """
    # Copie de la solution initiale
    solution = initial_solution.copy()

    # Calcul vectorisé du poids total pour chaque contrainte
    total_weights = np.dot(resource_consumption, solution)

    # Réparation des contraintes violées
    while np.any(total_weights > resource_availabilities):
        # Calcul des ratios profit / coût pour les projets actifs
        active_projects = solution == 1
        resource_costs = np.sum(resource_consumption[:, active_projects], axis=0)
        ratios = profits[active_projects] / resource_costs

        # Trouver le projet à retirer (plus petit ratio profit / coût)
        item_to_remove = np.where(active_projects)[0][np.argmin(ratios)]

        # Mise à jour de la solution
        solution[item_to_remove] = 0

        # Mise à jour des ressources consommées
        total_weights -= resource_consumption[:, item_to_remove]

    # Calcul du profit total
    total_profit = calculate_profit(solution, profits)

    return solution, total_profit