import os
import numpy as np

from utils import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from heuristics.repair_heuristic import repair_heuristic

import hill_climbing.neighborhoods as neighborhoods
from hill_climbing.hill_climbing import hill_climbing

from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic


def test_read_knapsack_data(instance_name):
    try:
        # Lire les données
        instances = read_knapsack_data(instance_name)

        # Affichage des données extraites
        print("Données extraites des instances :")
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1} :")
            print(f"  - Nombre de projets (N) : {instance['N']}")
            print(f"  - Nombre de ressources (M) : {instance['M']}")
            print(f"  - Valeur optimale : {instance['optimal_value']}")
            
            print("  - Profits :")
            print(instance['profits'])
            
            print("  - Consommation des ressources :")
            for resource_idx, resource_row in enumerate(instance['resource_consumption']):
                print(f"    Ressource {resource_idx + 1} : {resource_row}")
            
            print("  - Disponibilités des ressources :")
            print(instance['resource_availabilities'])

        print("\nTest réussi : les données ont été correctement lues.")
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture des données : {e}")


def test_greedy_heuristic(instance_name):
    # Lire les données
    try:
        data = read_knapsack_data(instance_name)
    except Exception as e:
        print(f"Erreur lors de la lecture des données : {e}")
        return

    # Tester l'heuristique sur chaque instance
    for i, instance in enumerate(data):
        print(f"\nInstance {i + 1}:")
        print(f"  - Nombre de projets (N): {instance['N']}")
        print(f"  - Nombre de ressources (M): {instance['M']}")
        print(f"  - Profit optimal: {instance['optimal_value']}")

        # Convertir les données en NumPy arrays
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        # Exécuter l'heuristique optimisée
        solution, total_profit = greedy_heuristic(
            instance['N'],
            instance['M'],
            profits,
            resource_consumption,
            resource_availabilities
        )

        # Résultats
        print(f"  - Solution trouvée: {solution}")
        print(f"  - Profit trouvé {total_profit}")


def test_repair_heuristic(instance_name):
    # Lire les données
    try:
        data = read_knapsack_data(instance_name)
    except Exception as e:
        print(f"Erreur lors de la lecture des données : {e}")
        return

    # Tester l'heuristique sur chaque instance
    for i, instance in enumerate(data):
        print(f"\nInstance {i + 1}:")
        print(f"  - Nombre de projets (N): {instance['N']}")
        print(f"  - Nombre de ressources (M): {instance['M']}")
        print(f"  - Profit optimal: {instance['optimal_value']}")

        # Convertir les données en NumPy arrays
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        # Exécuter l'heuristique de réparation avec la solution initiale issue de la relaxation surrogate
        repaired_solution, total_profit = repair_heuristic(
            instance['N'],
            instance['M'],
            resource_consumption,
            resource_availabilities,
            profits
        )

        # Résultats
        print(f"  - Solution réparée (faisable): {repaired_solution}")
        print(f"  - Profit trouvé: {total_profit}")


def test_hill_climbing(instance_name):
    try:
        # Lire les données
        data = read_knapsack_data(instance_name)
        
        print("Test de l'algorithme de montée de colline avec une solution initiale obtenue par greedy_heuristic:")

        # Tester l'algorithme sur chaque instance
        for i, instance in enumerate(data):
            print(f"\nInstance {i + 1}:")
            print(f"  - Nombre de projets (N): {instance['N']}")
            print(f"  - Nombre de ressources (M): {instance['M']}")
            print(f"  - Profit optimal: {instance['optimal_value']}")

            # Convertir les données en NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Exécuter l'heuristique greedy pour obtenir une solution initiale
            initial_solution, _ = greedy_heuristic(
                instance['N'],
                instance['M'],
                profits,
                resource_consumption,
                resource_availabilities
            )

            print(f"  - Solution initiale (obtenue par greedy): {initial_solution}")

            # Appliquer l'algorithme de montée de colline à partir de la solution initiale
            final_solution, total_profit = hill_climbing(
                instance['N'],
                instance['M'],
                initial_solution,
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood
            )

            # Afficher les résultats
            print(f"  - Solution finale après montée : {final_solution}")
            print(f"  - Profit final: {total_profit:.2f}")

            if instance['optimal_value'] is not None:
                print(f"  - Écart par rapport à l'optimal : {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest réussi : l'algorithme de montée a été exécuté sur toutes les instances.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de l'algorithme : {e}")


def test_simulated_annealing_metaheuristic(instance_name):
    try:
        # Lire les données
        instances = read_knapsack_data(instance_name)

        print("Test de l'algorithme de recuit simulé sur les instances :")

        # Tester l'algorithme sur chaque instance
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1} :")
            print(f"  - Nombre de projets (N) : {instance['N']}")
            print(f"  - Nombre de ressources (M) : {instance['M']}")
            print(f"  - Valeur optimale (si disponible) : {instance['optimal_value']}")

            # Convertir les données en NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Générer une solution initiale faisable (exemple : aucune sélection)
            initial_solution = np.zeros(instance['N'], dtype=np.int32)

            # Exécuter le recuit simulé
            final_solution, total_profit = simulated_annealing_metaheuristic(
                instance['N'],
                instance['M'],
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood,
                initial_temperature=1000,
                cooling_rate=0.95,
                max_iterations=1000,
                epsilon=1e-3
            )

            # Afficher les résultats
            print(f"  - Solution finale : {final_solution}")
            print(f"  - Profit final : {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Écart par rapport à l'optimal : {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest réussi : l'algorithme a été exécuté sur toutes les instances.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de l'algorithme : {e}")


def compare_methods(instance_name):
    try:
        # Lire les données
        data = read_knapsack_data(instance_name)

        print("Comparaison des méthodes sur les instances :")

        # Comparer les méthodes pour chaque instance
        for i, instance in enumerate(data):
            print(f"\nInstance {i + 1} :")
            print(f"  - Nombre de projets (N) : {instance['N']}")
            print(f"  - Nombre de ressources (M) : {instance['M']}")
            print(f"  - Valeur optimale (si disponible) : {instance['optimal_value']}")

            # Convertir les données en NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # 1. Méthode gloutonne
            greedy_solution, greedy_profit = greedy_heuristic(
                instance['N'],
                instance['M'],
                profits,
                resource_consumption,
                resource_availabilities
            )

            print(f"  - [Greedy] Profit : {greedy_profit:.2f}")

            # 2. Heuristique de réparation
            repair_solution, repair_profit = repair_heuristic(
                instance['N'],
                instance['M'],
                resource_consumption,
                resource_availabilities,
                profits
            )

            print(f"  - [Repair] Profit : {repair_profit:.2f}")

            # 3. Hill climbing
            hill_solution, hill_profit = hill_climbing(
                instance['N'],
                instance['M'],
                greedy_solution,  # Utilisation de la solution gloutonne comme point de départ
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood
            )

            print(f"  - [Hill Climbing] Profit : {hill_profit:.2f}")

            # 4. Recuit simulé
            annealing_solution, annealing_profit = simulated_annealing_metaheuristic(
                instance['N'],
                instance['M'],
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood,
                initial_temperature=1000,
                cooling_rate=0.95,
                max_iterations=10000,
                epsilon=1e-6
            )

            print(f"  - [Simulated Annealing] Profit : {annealing_profit:.2f}")

            # Comparaison avec la valeur optimale si disponible
            if instance['optimal_value'] is not None:
                print(f"  - Écart [Greedy] : {abs(instance['optimal_value'] - greedy_profit):.2f}")
                print(f"  - Écart [Repair] : {abs(instance['optimal_value'] - repair_profit):.2f}")
                print(f"  - Écart [Hill Climbing] : {abs(instance['optimal_value'] - hill_profit):.2f}")
                print(f"  - Écart [Simulated Annealing] : {abs(instance['optimal_value'] - annealing_profit):.2f}")

        print("\nComparaison terminée avec succès.")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution des comparaisons : {e}")


if __name__ == "__main__":
    test_read_knapsack_data("mknapcb1")