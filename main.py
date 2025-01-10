import os
import numpy as np

from read_knapsack_data import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from heuristics.repair_heuristic import repair_heuristic

import hill_climbing.neighbours as neighbours
from hill_climbing.hill_climbing import hill_climbing

from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic


def test_read_knapsack_data():
    # Chemin vers le fichier d'instances
    test_file_path = os.path.join("instances", "mknap1.txt")

    if not os.path.exists(test_file_path):
        print(f"Fichier de test non trouvé : {test_file_path}")
        return

    try:
        # Lire les données
        instances = read_knapsack_data(test_file_path)

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


def test_greedy_heuristic():
    # Charger les instances
    test_file_path = os.path.join("instances", "mknap1.txt")

    if not os.path.exists(test_file_path):
        print(f"Fichier de test non trouvé : {test_file_path}")
        return

    # Lire les données
    try:
        data = read_knapsack_data(test_file_path)
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


def test_repair_heuristic():
    # Charger les instances
    test_file_path = os.path.join("instances", "mknap1.txt")

    if not os.path.exists(test_file_path):
        print(f"Fichier de test non trouvé : {test_file_path}")
        return

    # Lire les données
    try:
        data = read_knapsack_data(test_file_path)
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

        # Générer une solution initiale irréalisable (tous les projets sélectionnés)
        initial_solution = np.ones(instance['N'], dtype=np.int32)

        # Exécuter l'heuristique de réparation
        repaired_solution, total_profit = repair_heuristic(
            instance['N'],
            instance['M'],
            initial_solution,
            resource_consumption,
            resource_availabilities,
            profits
        )

        # Résultats
        print(f"  - Solution initiale (non faisable): {initial_solution}")
        print(f"  - Solution réparée (faisable): {repaired_solution}")
        print(f"  - Profit trouvé: {total_profit}")

def test_simulated_annealing_metaheuristic():
    # Chemin vers le fichier d'instances
    test_file_path = os.path.join("instances", "mknap1.txt")

    if not os.path.exists(test_file_path):
        print(f"Fichier de test non trouvé : {test_file_path}")
        return

    try:
        # Lire les données
        instances = read_knapsack_data(test_file_path)

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
                initial_solution,
                resource_consumption,
                resource_availabilities,
                profits,
                initial_temperature=1000,
                cooling_rate=0.95,
                max_iterations=1000
            )

            # Afficher les résultats
            print(f"  - Solution finale : {final_solution}")
            print(f"  - Profit final : {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Écart par rapport à l'optimal : {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest réussi : l'algorithme a été exécuté sur toutes les instances.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de l'algorithme : {e}")


def test_hill_climbing():
    # Charger les instances
    test_file_path = os.path.join("instances", "mknap1.txt")

    if not os.path.exists(test_file_path):
        print(f"Fichier de test non trouvé : {test_file_path}")
        return

    try:
        # Lire les données
        data = read_knapsack_data(test_file_path)
        
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
                neighbours.three_opt_neighborhood
            )

            # Afficher les résultats
            print(f"  - Solution finale après montée : {final_solution}")
            print(f"  - Profit final: {total_profit:.2f}")

            if instance['optimal_value'] is not None:
                print(f"  - Écart par rapport à l'optimal : {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest réussi : l'algorithme de montée a été exécuté sur toutes les instances.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de l'algorithme : {e}")


if __name__ == "__main__":
    test_read_knapsack_data()
    test_hill_climbing()