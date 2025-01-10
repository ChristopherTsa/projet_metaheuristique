import os
from read_knapsack_data import read_knapsack_data


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


if __name__ == "__main__":
    test_read_knapsack_data()