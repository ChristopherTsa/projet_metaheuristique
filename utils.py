import numpy as np
import pandas as pd


def read_knapsack_data(file_path):
    """
    Reads a multidimensional knapsack data file and extracts the instances.

    Args:
        file_path (str): Path to the data file.

    Returns:
        list: A list of dictionaries, each representing an instance with the following keys:
            - 'N': Number of projects (int)
            - 'M': Number of resources (int)
            - 'optimal_value': Optimal value for the instance (float or None)
            - 'profits': List of profits (list of floats)
            - 'resource_consumption': List of lists where each sublist contains resource consumption for each project
            - 'resource_availabilities': List of available quantities of resources (list of floats)
    """
    instances = []

    with open(file_path, 'r') as file:
        # Supprimer les lignes vides
        lines = [line.strip() for line in file if line.strip()]
        total_lines = len(lines)
        # Première ligne : nombre d'instances
        try:
            num_instances = int(lines[0])
        except ValueError:
            raise ValueError("La première ligne doit spécifier le nombre d'instances sous forme d'entier.")

        idx = 1  # index de la ligne actuelle

        for instance_idx in range(num_instances):
            if idx >= total_lines:
                raise ValueError(f"Données manquantes pour l'instance {instance_idx + 1}.")

            # Lire les paramètres de l'instance
            try:
                instance_header = list(map(float, lines[idx].split()))
                N, M, optimal_value = int(instance_header[0]), int(instance_header[1]), instance_header[2]
                idx += 1
            except (IndexError, ValueError) as e:
                raise ValueError(f"Erreur de lecture des paramètres pour l'instance {instance_idx + 1} à la ligne {idx}: {e}")

            # Lire les profits
            profits = []
            while len(profits) < N:
                if idx >= total_lines:
                    raise ValueError(f"Données insuffisantes pour les profits de l'instance {instance_idx + 1}.")
                profits_line = list(map(float, lines[idx].split()))
                profits.extend(profits_line)
                idx += 1
            if len(profits) > N:
                raise ValueError(f"Trop de valeurs de profits pour l'instance {instance_idx + 1}.")

            # Lire la consommation des ressources
            resource_consumption = []
            for resource_idx in range(M):
                resource_row = []
                while len(resource_row) < N:
                    if idx >= total_lines:
                        raise ValueError(f"Données insuffisantes pour la consommation de ressource {resource_idx + 1} de l'instance {instance_idx + 1}.")
                    consumption_line = list(map(float, lines[idx].split()))
                    resource_row.extend(consumption_line)
                    idx += 1
                if len(resource_row) > N:
                    raise ValueError(f"Trop de valeurs de consommation pour la ressource {resource_idx + 1} de l'instance {instance_idx + 1}.")
                resource_consumption.append(resource_row)

            # Lire les quantités de ressources disponibles
            resource_availabilities = []
            while len(resource_availabilities) < M:
                if idx >= total_lines:
                    raise ValueError(f"Données insuffisantes pour les ressources disponibles de l'instance {instance_idx + 1}.")
                availabilities_line = list(map(float, lines[idx].split()))
                resource_availabilities.extend(availabilities_line)
                idx += 1
            if len(resource_availabilities) > M:
                raise ValueError(f"Trop de valeurs pour les ressources disponibles de l'instance {instance_idx + 1}.")

            # Stocker les données de l'instance
            instances.append({
                'N': N,
                'M': M,
                'optimal_value': optimal_value if optimal_value != 0 else None,
                'profits': profits,
                'resource_consumption': resource_consumption,
                'resource_availabilities': resource_availabilities
            })

    return instances


def calculate_profit(solution, profits):
    """Calculate the total profit of a solution."""
    return np.sum(profits * solution)


def is_feasible(solution, M, resource_consumption, resource_availabilities):
    """Check if a solution is feasible."""
    for i in range(M):
        if np.sum(resource_consumption[i, :] * solution) > resource_availabilities[i]:
            return False
    return True