import os
import time
import numpy as np
import pandas as pd

from utilities import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from heuristics.repair_heuristic import repair_heuristic, surrogate_relaxation_mkp

import hill_climbing.neighborhoods as neighborhoods
from hill_climbing.hill_climbing import hill_climbing
from hill_climbing.vns import vns_hill_climbing

from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic
from metaheuristic.genetic_metaheuristic import genetic_metaheuristic
from metaheuristic.sa_iga_metaheuristic import sa_iga_metaheuristic


def test_read_knapsack_data(instance_name):
    """
    Test function for reading and displaying data from knapsack problem instances.

    Args:
        instance_name (str): Name of the instance file to be read.

    Returns:
        None
    """
    try:
        # Read knapsack data from the instance file
        instances = read_knapsack_data(instance_name)

        # Display extracted data
        print("Extracted data from instances:")
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal value: {instance['optimal_value']}")

            print("  - Profits:")
            print(instance['profits'])

            print("  - Resource consumption:")
            for resource_idx, resource_row in enumerate(instance['resource_consumption']):
                print(f"    Resource {resource_idx + 1}: {resource_row}")

            print("  - Resource availabilities:")
            print(instance['resource_availabilities'])

        print("\nTest successful: Data has been read correctly.")
    except Exception as e:
        print(f"An error occurred while reading data: {e}")


def test_greedy_heuristic(instance_name):
    """
    Test function for the greedy heuristic algorithm.

    Args:
        instance_name (str): Name of the instance file to be read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        data = read_knapsack_data(instance_name)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Apply the greedy heuristic on each instance
    for i, instance in enumerate(data):
        print(f"\nInstance {i + 1}:")
        print(f"  - Number of projects (N): {instance['N']}")
        print(f"  - Number of resources (M): {instance['M']}")
        print(f"  - Optimal profit: {instance['optimal_value']}")

        # Convert data to NumPy arrays
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        # Execute the greedy heuristic
        solution, total_profit = greedy_heuristic(
            instance['N'],
            profits,
            resource_consumption,
            resource_availabilities
        )

        # Display results
        print(f"  - Solution found: {solution}")
        print(f"  - Profit found: {total_profit}")


def test_repair_heuristic(instance_name):
    """
    Test function for the repair heuristic algorithm.

    Args:
        instance_name (str): Name of the instance file to be read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        data = read_knapsack_data(instance_name)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Apply the repair heuristic on each instance
    for i, instance in enumerate(data):
        print(f"\nInstance {i + 1}:")
        print(f"  - Number of projects (N): {instance['N']}")
        print(f"  - Number of resources (M): {instance['M']}")
        print(f"  - Optimal profit: {instance['optimal_value']}")

        # Convert data to NumPy arrays
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        # Execute the repair heuristic
        initial_solution = surrogate_relaxation_mkp(
            instance['N'], resource_consumption, resource_availabilities, profits
        )
        repaired_solution, total_profit = repair_heuristic(
            initial_solution,
            resource_consumption,
            resource_availabilities,
            profits
        )

        # Display results
        print(f"  - Repaired solution (feasible): {repaired_solution}")
        print(f"  - Profit found: {total_profit}")


def test_hill_climbing(instance_name):
    """
    Test function for the hill climbing algorithm.

    Args:
        instance_name (str): Name of the instance file to be read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        data = read_knapsack_data(instance_name)

        print("Testing hill climbing algorithm with an initial solution from the greedy heuristic:")

        # Apply hill climbing on each instance
        for i, instance in enumerate(data):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal profit: {instance['optimal_value']}")

            # Convert data to NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Get an initial solution from the greedy heuristic
            initial_solution, _ = greedy_heuristic(
                instance['N'],
                profits,
                resource_consumption,
                resource_availabilities
            )
            print(f"  - Initial solution (greedy): {initial_solution}")

            # Apply hill climbing
            final_solution, total_profit = hill_climbing(
                instance['N'],
                initial_solution,
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.resource_profit_based_1_pair_neighborhood
            )

            # Display results
            print(f"  - Final solution after hill climbing: {final_solution}")
            print(f"  - Final profit: {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Deviation from optimal: {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest successful: Hill climbing executed on all instances.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")


def test_vns_hill_climbing(instance_name):
    """
    Test function for the VNS Hill Climbing algorithm on knapsack instances.

    Args:
        instance_name (str): Name of the instance file to read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        instances = read_knapsack_data(instance_name)

        print("Testing the VNS Hill Climbing algorithm on instances:")

        # Apply the algorithm to each instance
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal value (if available): {instance['optimal_value']}")

            # Convert data to NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Execute the Greedy heuristic to get an initial solution
            initial_solution, _ = greedy_heuristic(
                instance['N'],
                profits,
                resource_consumption,
                resource_availabilities
            )
            print(f"  - Initial solution (Greedy): {initial_solution}")

            # Execute the VNS Hill Climbing algorithm
            final_solution, total_profit = vns_hill_climbing(
                instance['N'],
                initial_solution,
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood,
                max_time=60,  # Maximum execution time in seconds
                k_max=3       # Maximum neighborhood size
            )

            # Display results
            print(f"  - Final solution: {final_solution}")
            print(f"  - Final profit: {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Deviation from optimal: {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest successful: VNS Hill Climbing executed on all instances.")
    except Exception as e:
        print(f"An error occurred while executing the algorithm: {e}")


def test_simulated_annealing_metaheuristic(instance_name):
    """
    Test function for the Simulated Annealing metaheuristic on knapsack instances.

    Args:
        instance_name (str): Name of the instance file to read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        instances = read_knapsack_data(instance_name)

        print("Testing the Simulated Annealing algorithm on instances:")

        # Apply the algorithm to each instance
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal value (if available): {instance['optimal_value']}")

            # Convert data to NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Execute the Simulated Annealing algorithm
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

            # Display results
            print(f"  - Final solution: {final_solution}")
            print(f"  - Final profit: {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Deviation from optimal: {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest successful: Simulated Annealing executed on all instances.")
    except Exception as e:
        print(f"An error occurred while executing the algorithm: {e}")


def test_genetic_algorithm(instance_name):
    """
    Test function for the Genetic Algorithm on knapsack instances.

    Args:
        instance_name (str): Name of the instance file to read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        instances = read_knapsack_data(instance_name)

        print("Testing the Genetic Algorithm on instances:")

        # Apply the algorithm to each instance
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal value (if available): {instance['optimal_value']}")

            # Convert data to NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Execute the Genetic Algorithm
            final_solution, total_profit = genetic_metaheuristic(
                instance['N'],
                instance['M'],
                resource_consumption,
                resource_availabilities,
                profits
            )

            # Display results
            print(f"  - Final solution: {final_solution}")
            print(f"  - Final profit: {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Deviation from optimal: {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest successful: Genetic Algorithm executed on all instances.")
    except Exception as e:
        print(f"An error occurred while executing the algorithm: {e}")


def test_sa_iga(instance_name):
    """
    Test function for the SA IGA (Simulated Annealing + Genetic Algorithm) on knapsack instances.

    Args:
        instance_name (str): Name of the instance file to read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        instances = read_knapsack_data(instance_name)

        print("Testing the SA IGA algorithm on instances:")

        # Apply the algorithm to each instance
        for i, instance in enumerate(instances):
            print(f"\nInstance {i + 1}:")
            print(f"  - Number of projects (N): {instance['N']}")
            print(f"  - Number of resources (M): {instance['M']}")
            print(f"  - Optimal value (if available): {instance['optimal_value']}")

            # Convert data to NumPy arrays
            profits = np.array(instance['profits'], dtype=np.float64)
            resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
            resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

            # Execute the SA IGA algorithm
            final_solution, total_profit = sa_iga_metaheuristic(
                instance['N'],
                instance['M'],
                resource_consumption,
                resource_availabilities,
                profits,
                neighborhoods.multi_opt_neighborhood,  # Neighborhood function
                initial_temperature=1000,  # Initial temperature for SA
                cooling_rate=0.75,        # Cooling rate for SA
                max_iterations_sa=10,     # Maximum iterations for SA
                epsilon=1e-3,             # Convergence threshold for SA
                population_size=100,      # Population size for GA
                ngen=20,                  # Number of generations for GA
                cxpb=0.7,                 # Crossover probability for GA
                mutpb=0.3                 # Mutation probability for GA
            )

            # Display results
            print(f"  - Final solution: {final_solution}")
            print(f"  - Final profit: {total_profit:.2f}")
            if instance['optimal_value'] is not None:
                print(f"  - Deviation from optimal: {abs(instance['optimal_value'] - total_profit):.2f}")

        print("\nTest successful: SA IGA executed on all instances.")
    except Exception as e:
        print(f"An error occurred while executing the algorithm: {e}")


def compare_methods(instance_name, max_instances=5):
    """
    Compares multiple methods (heuristics and metaheuristics) for solving the knapsack problem on given instances.

    Args:
        instance_name (str): Name of the instance file to read.

    Returns:
        None
    """
    try:
        # Read knapsack data
        data = read_knapsack_data(instance_name)

        print("Comparing methods on instances:")

        # Define output CSV file and column names
        output_file = os.path.join("results", f"{instance_name}.csv")
        results_columns = [
            "Instance", "Number of projects (N)", "Number of resources (M)", "Optimal profit",
            "[Greedy] Profit", "[Repair] Profit", "[Hill Climbing] Profit", "[VNS + Hill Climbing] Profit",
            "[SA] Profit", 
            # "[Genetic] Profit", "[SA IGA] Profit",  # Commented methods
            "Deviation [Greedy]", "Deviation [Repair]", "Deviation [Hill Climbing]", 
            "Deviation [VNS + Hill Climbing]", "Deviation [SA]", 
            # "Deviation [Genetic]", "Deviation [SA IGA]",  # Commented methods
            "Time [Greedy]", "Time [Repair]", "Time [Hill Climbing]", 
            "Time [VNS + Hill Climbing]", "Time [SA]" 
            # "Time [Genetic]", "Time [SA IGA]"  # Commented methods
        ]

        # Initialize the CSV file if it doesn't exist
        if not os.path.exists(output_file):
            pd.DataFrame(columns=results_columns).to_csv(output_file, index=False)

        # Compare methods for each instance
        for i, instance in enumerate(data):
            if i < max_instances:  # Limit the number of instances to process
                print(f"\nInstance {i + 1}:")
                print(f"  - Number of projects (N): {instance['N']}")
                print(f"  - Number of resources (M): {instance['M']}")
                print(f"  - Optimal profit (if available): {instance['optimal_value']}")

                # Convert data to NumPy arrays
                profits = np.array(instance['profits'], dtype=np.float64)
                resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
                resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

                # 1. Greedy heuristic
                start_time_greedy = time.time()
                greedy_solution, greedy_profit = greedy_heuristic(
                    instance['N'],
                    profits,
                    resource_consumption,
                    resource_availabilities
                )
                end_time_greedy = time.time()
                greedy_time = end_time_greedy - start_time_greedy
                print(f"  - [Greedy] Profit: {greedy_profit:.2f}")

                # 2. Repair heuristic
                #start_time_repair = time.time()
                #initial_solution = surrogate_relaxation_mkp(
                #    instance['N'], resource_consumption, resource_availabilities, profits
                #)
                #repair_solution, repair_profit = repair_heuristic(
                #    initial_solution,
                #    resource_consumption,
                #    resource_availabilities,
                #    profits
                #)
                #end_time_repair = time.time()
                #repair_time = end_time_repair - start_time_repair
                #print(f"  - [Repair] Profit: {repair_profit:.2f}")

                # 3. Hill Climbing
                #start_time_hill_climbing = time.time()
                #hill_solution, hill_profit = hill_climbing(
                #    instance['N'],
                #    greedy_solution,  # Use greedy solution as starting point
                #    resource_consumption,
                #    resource_availabilities,
                #    profits,
                #    neighborhoods.multi_opt_neighborhood,
                #    3
                #)
                #end_time_hill_climbing = time.time()
                #hill_climbing_time = end_time_hill_climbing - start_time_hill_climbing
                #print(f"  - [Hill Climbing] Profit: {hill_profit:.2f}")

                # 4. VNS + Hill Climbing
                start_time_vns = time.time()
                vns_solution, vns_profit = vns_hill_climbing(
                    instance['N'],
                    greedy_solution,  # Use greedy solution as starting point
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    neighborhoods.multi_opt_neighborhood,
                    60, # Maximum duration
                    2 # Neighborhood degree
                )
                end_time_vns = time.time()
                vns_time = end_time_vns - start_time_vns
                print(f"  - [VNS + Hill Climbing] Profit: {vns_profit:.2f}")

                # 5. Simulated Annealing
                #start_time_sa = time.time()
                #sa_solution, sa_profit = simulated_annealing_metaheuristic(
                #    instance['N'],
                #    resource_consumption,
                #    resource_availabilities,
                #    profits,
                #    neighborhoods.multi_opt_neighborhood,
                #    greedy_solution,  # Use greedy solution as starting point
                #    300,  # Maximum duration
                #    100,  # Maximum iterations
                #    None, # Initial temperature
                #    0.95,  # Cooling rate
                #    1e-5,  # Convergence threshold
                #    2   # Neighborhood degree
                #)
                #end_time_sa = time.time()
                #sa_time = end_time_sa - start_time_sa
                #print(f"  - [SA] Profit: {sa_profit:.2f}")

                # 6. Genetic Algorithm
                start_time_genetic = time.time()
                genetic_solution, genetic_profit = genetic_metaheuristic(
                    instance['N'],
                    instance['M'],
                    resource_consumption,
                    resource_availabilities,
                    profits
                )
                end_time_genetic = time.time()
                genetic_time = end_time_genetic - start_time_genetic
                print(f"  - [Genetic] Profit: {genetic_profit:.2f}")

                # 7. SA IGA
                # start_time_sa_iga = time.time()
                # sa_iga_solution, sa_iga_profit = sa_iga_metaheuristic(
                #     instance['N'],
                #     instance['M'],
                #     resource_consumption,
                #     resource_availabilities,
                #     profits,
                #     neighborhoods.multi_opt_neighborhood
                # )
                # end_time_sa_iga = time.time()
                # sa_iga_time = end_time_sa_iga - start_time_sa_iga
                # print(f"  - [SA IGA] Profit: {sa_iga_profit:.2f}")

                # Calculate deviations if optimal value is available
                if instance['optimal_value'] is not None:
                    print(f"  - Deviation [Greedy]: {abs(instance['optimal_value'] - greedy_profit):.2f}")
                    # print(f"  - Deviation [Repair]: {abs(instance['optimal_value'] - repair_profit):.2f}")
                    # print(f"  - Deviation [Hill Climbing]: {abs(instance['optimal_value'] - hill_profit):.2f}")
                    print(f"  - Deviation [VNS + Hill Climbing]: {abs(instance['optimal_value'] - vns_profit):.2f}")
                    # print(f"  - Deviation [SA]: {abs(instance['optimal_value'] - sa_profit):.2f}")
                    print(f"  - Deviation [Genetic]: {abs(instance['optimal_value'] - genetic_profit):.2f}")
                    # print(f"  - Deviation [SA IGA]: {abs(instance['optimal_value'] - sa_iga_profit):.2f}")

                # Save results to the CSV file
                result = {
                    "Instance": i + 1,
                    "Number of projects (N)": instance['N'],
                    "Number of resources (M)": instance['M'],
                    "Optimal profit": instance['optimal_value'],
                    # "[Greedy] Profit": greedy_profit,
                    # "[Repair] Profit": repair_profit,
                    # "[Hill Climbing] Profit": hill_profit,
                    "[VNS + Hill Climbing] Profit": vns_profit,
                    # "[SA] Profit": sa_profit,
                    "[Genetic] Profit": genetic_profit,
                    # "[SA IGA] Profit": sa_iga_profit,
                    # "Deviation [Greedy]": abs(instance['optimal_value'] - greedy_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [Repair]": abs(instance['optimal_value'] - repair_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [Hill Climbing]": abs(instance['optimal_value'] - hill_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [VNS + Hill Climbing]": abs(instance['optimal_value'] - vns_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [SA]": abs(instance['optimal_value'] - sa_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [Genetic]": abs(instance['optimal_value'] - genetic_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [SA IGA]": abs(instance['optimal_value'] - sa_iga_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Time [Greedy]": greedy_time,
                    # "Time [Repair]": repair_time,
                    # "Time [Hill Climbing]": hill_climbing_time,
                    "Time [VNS + Hill Climbing]": vns_time,
                    #"Time [SA]": sa_time,
                    "Time [Genetic]": genetic_time,
                    # "Time [SA IGA]": sa_iga_time
                }

                # Append results to the CSV file
                pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False)

        print("\nComparison completed successfully. Results saved in:", output_file)

    except Exception as e:
        print(f"An error occurred during the comparisons: {e}")