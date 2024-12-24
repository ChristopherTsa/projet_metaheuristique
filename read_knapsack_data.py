import os
import numpy as np

def read_knapsack_data(file_path):
    """
    Reads a multidimensional knapsack data file and extracts the instances.

    Args:
        file_path (str): Path to the data file.

    Returns:
        list: A list of dictionaries, each representing an instance with the following keys:
            - 'N': Number of projects (int)
            - 'M': Number of resources (int)
            - 'optimal_value': Optimal value for the instance (int or None)
            - 'profits': List of profits (list of ints)
            - 'resource_consumption': List of lists where each sublist contains resource consumption for each project
            - 'resource_availabilities': List of available quantities of resources (list of ints)
    """
    instances = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # First line: number of instances
        num_instances = int(lines[0].strip())
        idx = 1

        for _ in range(num_instances):
            # Read instance parameters
            instance_header = list(map(int, lines[idx].strip().split()))
            N, M, optimal_value = instance_header[0], instance_header[1], instance_header[2]
            idx += 1

            # Read profits
            profits = list(map(int, lines[idx].strip().split()))
            idx += 1

            # Read resource consumption for each resource
            resource_consumption = []
            for _ in range(M):
                resource_row = list(map(int, lines[idx].strip().split()))
                resource_consumption.append(resource_row)
                idx += 1

            # Read resource availabilities
            resource_availabilities = list(map(int, lines[idx].strip().split()))
            idx += 1

            # Store the instance data
            instances.append({
                'N': N,
                'M': M,
                'optimal_value': optimal_value if optimal_value != 0 else None,
                'profits': profits,
                'resource_consumption': resource_consumption,
                'resource_availabilities': resource_availabilities
            })

    return instances

# Example usage:
# data = read_mknap_data("mknap1.txt")
# for instance in data:
#     print(instance)
