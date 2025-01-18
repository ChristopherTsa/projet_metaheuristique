import os
import time
import numpy as np
import pandas as pd

import test

from utilities import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from heuristics.repair_heuristic import repair_heuristic, surrogate_relaxation_mkp

import hill_climbing.neighborhoods as neighborhoods
from hill_climbing.hill_climbing import hill_climbing
from hill_climbing.vns import vns_hill_climbing

from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic
from metaheuristic.genetic_metaheuristic import genetic_metaheuristic
from metaheuristic.sa_iga_metaheuristic import sa_iga_metaheuristic


def compare_methods(instance_name):
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
            if i < 10:  # Limit the number of instances to process
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
                start_time_repair = time.time()
                initial_solution = surrogate_relaxation_mkp(
                    instance['N'], resource_consumption, resource_availabilities, profits
                )
                repair_solution, repair_profit = repair_heuristic(
                    initial_solution,
                    resource_consumption,
                    resource_availabilities,
                    profits
                )
                end_time_repair = time.time()
                repair_time = end_time_repair - start_time_repair
                print(f"  - [Repair] Profit: {repair_profit:.2f}")

                # 3. Hill Climbing
                start_time_hill_climbing = time.time()
                hill_solution, hill_profit = hill_climbing(
                    instance['N'],
                    greedy_solution,  # Use greedy solution as starting point
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    neighborhoods.multi_opt_neighborhood,
                    3
                )
                end_time_hill_climbing = time.time()
                hill_climbing_time = end_time_hill_climbing - start_time_hill_climbing
                print(f"  - [Hill Climbing] Profit: {hill_profit:.2f}")

                # 4. VNS + Hill Climbing
                start_time_vns = time.time()
                vns_solution, vns_profit = vns_hill_climbing(
                    instance['N'],
                    greedy_solution,  # Use greedy solution as starting point
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    neighborhoods.multi_opt_neighborhood,
                    1,
                    3
                )
                end_time_vns = time.time()
                vns_time = end_time_vns - start_time_vns
                print(f"  - [VNS + Hill Climbing] Profit: {vns_profit:.2f}")

                # 5. Simulated Annealing
                start_time_sa = time.time()
                sa_solution, sa_profit = simulated_annealing_metaheuristic(
                    instance['N'],
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    neighborhoods.multi_opt_neighborhood,
                    greedy_solution,  # Use greedy solution as starting point
                    60,  # Initial temperature
                    100,  # Maximum iterations
                    None,
                    0.95,  # Cooling rate
                    1e-5,  # Convergence threshold
                    3
                )
                end_time_sa = time.time()
                sa_time = end_time_sa - start_time_sa
                print(f"  - [SA] Profit: {sa_profit:.2f}")

                # 6. Genetic Algorithm
                # start_time_genetic = time.time()
                # genetic_solution, genetic_profit = genetic_metaheuristic(
                #     instance['N'],
                #     instance['M'],
                #     resource_consumption,
                #     resource_availabilities,
                #     profits
                # )
                # end_time_genetic = time.time()
                # genetic_time = end_time_genetic - start_time_genetic
                # print(f"  - [Genetic] Profit: {genetic_profit:.2f}")

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
                    print(f"  - Deviation [Repair]: {abs(instance['optimal_value'] - repair_profit):.2f}")
                    print(f"  - Deviation [Hill Climbing]: {abs(instance['optimal_value'] - hill_profit):.2f}")
                    print(f"  - Deviation [VNS + Hill Climbing]: {abs(instance['optimal_value'] - vns_profit):.2f}")
                    print(f"  - Deviation [SA]: {abs(instance['optimal_value'] - sa_profit):.2f}")
                    # print(f"  - Deviation [Genetic]: {abs(instance['optimal_value'] - genetic_profit):.2f}")
                    # print(f"  - Deviation [SA IGA]: {abs(instance['optimal_value'] - sa_iga_profit):.2f}")

                # Save results to the CSV file
                result = {
                    "Instance": i + 1,
                    "Number of projects (N)": instance['N'],
                    "Number of resources (M)": instance['M'],
                    "Optimal profit": instance['optimal_value'],
                    "[Greedy] Profit": greedy_profit,
                    "[Repair] Profit": repair_profit,
                    "[Hill Climbing] Profit": hill_profit,
                    "[VNS + Hill Climbing] Profit": vns_profit,
                    "[SA] Profit": sa_profit,
                    # "[Genetic] Profit": genetic_profit,  # Commented
                    # "[SA IGA] Profit": sa_iga_profit,  # Commented
                    "Deviation [Greedy]": abs(instance['optimal_value'] - greedy_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [Repair]": abs(instance['optimal_value'] - repair_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [Hill Climbing]": abs(instance['optimal_value'] - hill_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [VNS + Hill Climbing]": abs(instance['optimal_value'] - vns_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Deviation [SA]": abs(instance['optimal_value'] - sa_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [Genetic]": abs(instance['optimal_value'] - genetic_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    # "Deviation [SA IGA]": abs(instance['optimal_value'] - sa_iga_profit) / instance['optimal_value'] if instance['optimal_value'] else None,
                    "Time [Greedy]": greedy_time,
                    "Time [Repair]": repair_time,
                    "Time [Hill Climbing]": hill_climbing_time,
                    "Time [VNS + Hill Climbing]": vns_time,
                    "Time [SA]": sa_time,
                    # "Time [Genetic]": genetic_time,  # Commented
                    # "Time [SA IGA]": sa_iga_time  # Commented
                }

                # Append results to the CSV file
                pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False)

        print("\nComparison completed successfully. Results saved in:", output_file)

    except Exception as e:
        print(f"An error occurred during the comparisons: {e}")


if __name__ == "__main__":
    compare_methods("mknap1")
