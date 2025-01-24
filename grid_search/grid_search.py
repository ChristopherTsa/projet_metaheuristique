import itertools
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hill_climbing.neighborhoods as n

from utilities import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from hill_climbing.hill_climbing import hill_climbing
from hill_climbing.vns import vns_hill_climbing
from metaheuristic.genetic_metaheuristic import genetic_metaheuristic
from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def grid_search_hc(data, instance_name, parameter_grid, max_instances=5):
    """
    Performs a grid search over hyperparameters for the Hill Climbing algorithm.

    Args:
        instance_name (str): Name of the instance file to read.
        parameter_grid (dict): Dictionary of Hill Climbing hyperparameters and their potential values.
        max_instances (int): Maximum number of instances to process.

    Returns:
        pd.DataFrame: A DataFrame with results for all parameter combinations.
    """
    parameter_combinations = list(itertools.product(*parameter_grid.values()))
    results = []
    output_dir = os.path.join("grid_search", "hill_climbing")
    ensure_directory_exists(output_dir)

    for i, instance in enumerate(data):
        if i >= max_instances:
            break

        print(f"\nProcessing instance {i + 1} for Hill Climbing...")
        
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        greedy_solution, _ = greedy_heuristic(
            instance['N'],
            profits,
            resource_consumption,
            resource_availabilities
        )

        for combination in parameter_combinations:
            params = dict(zip(parameter_grid.keys(), combination))
            hc_neighbors = params.get('hc_neighbors', n.multi_opt_neighborhood)
            hc_k = params.get('hc_k', 3)
            
            print(f"Processing instance {i + 1} with parameters: {params}")

            try:
                start_time = time.time()
                hc_solution, hc_profit = hill_climbing(
                    instance['N'],
                    greedy_solution,
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    generate_neighbors=hc_neighbors,
                    k=hc_k
                )
                hc_time = time.time() - start_time

                results.append({
                    'Instance': i + 1,
                    'hc_neighbors': hc_neighbors.__name__,
                    'hc_k': hc_k,
                    'HC Profit': hc_profit,
                    'HC Time': hc_time,
                    'HC Ecart': abs(instance['optimal_value'] - hc_profit)
                })
            except Exception as e:
                print(f"Error with parameters {params}: {e}")

    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{instance_name}.csv")
    results_df.to_csv(output_file, index=False)

    print("\nHill Climbing grid search completed. Results saved to:", output_dir)
    return results_df


def grid_search_ga(data, instance_name, parameter_grid, max_instances=5):
    """
    Performs a grid search over hyperparameters for the Genetic Algorithm.

    Args:
        instance_name (str): Name of the instance file to read.
        parameter_grid (dict): Dictionary of Genetic Algorithm hyperparameters and their potential values.
        max_instances (int): Maximum number of instances to process.

    Returns:
        pd.DataFrame: A DataFrame with results for all parameter combinations.
    """
    parameter_combinations = list(itertools.product(*parameter_grid.values()))
    results = []
    output_dir = os.path.join("grid_search", "genetic")
    ensure_directory_exists(output_dir)

    for i, instance in enumerate(data):
        if i >= max_instances:
            break

        print(f"\nProcessing instance {i + 1} for Genetic Algorithm...")
        
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)

        for combination in parameter_combinations:
            params = dict(zip(parameter_grid.keys(), combination))
            pop_size = params.get('pop_size', 200)
            cxpb = params.get('cxpb', 0.7)
            mutpb = params.get('mutpb', 0.2)
            ngen = params.get('ngen', 150)
            uniform_crossover_prob = params.get('uniform_crossover_prob', 0.5)
            bitflip_mutation_prob = params.get('bitflip_mutation_prob', 0.2)
            tournament_size = params.get('tournament_size', 3)
            
            print(f"Processing instance {i + 1} with parameters: {params}")

            try:
                start_time = time.time()
                ga_solution, ga_profit = genetic_metaheuristic(
                    instance['N'],
                    instance['M'],
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    popsize=pop_size,
                    cxpb=cxpb,
                    mutpb=mutpb,
                    ngen=ngen,
                    uniform_crossover_prob=uniform_crossover_prob,
                    bitflip_mutation_prob=bitflip_mutation_prob,
                    tournament_size=tournament_size
                )
                ga_time = time.time() - start_time

                results.append({
                    'Instance': i + 1,
                    'pop_size': pop_size,
                    'cxpb': cxpb,
                    'mutpb': mutpb,
                    'ngen': ngen,
                    'uniform_crossover_prob': uniform_crossover_prob,
                    'bitflip_mutation_prob': bitflip_mutation_prob,
                    'tournament_size': tournament_size,
                    'GA Profit': ga_profit,
                    'GA Time': ga_time,
                    'GA Ecart': abs(instance['optimal_value'] - ga_profit)
                })
            except Exception as e:
                print(f"Error with parameters {params}: {e}")

    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{instance_name}.csv")
    results_df.to_csv(output_file, index=False)

    print("\nGenetic Algorithm grid search completed. Results saved to:", output_dir)
    return results_df


def grid_search_vns(data, instance_name, parameter_grid, max_instances=5):
    """
    Performs a grid search over hyperparameters for the VNS heuristic.

    Args:
        instance_name (str): Name of the instance file to read.
        parameter_grid (dict): Dictionary of VNS hyperparameters and their potential values.
        max_instances (int): Maximum number of instances to process.

    Returns:
        pd.DataFrame: A DataFrame with results for all parameter combinations.
    """
    # Generate all combinations of hyperparameters
    parameter_combinations = list(itertools.product(*parameter_grid.values()))

    # Create a DataFrame to store results
    results = []

    # Ensure output directory exists
    output_dir = os.path.join("grid_search", "vns")
    ensure_directory_exists(output_dir)

    # Iterate through instances
    for i, instance in enumerate(data):
        if i >= max_instances:
            break

        print(f"\nProcessing instance {i + 1} for VNS...")
        
        # Prepare instance data
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)
        
        # Generate an initial solution using the greedy heuristic
        greedy_solution, greedy_profit = greedy_heuristic(
                instance['N'],
                profits,
                resource_consumption,
                resource_availabilities
            )

        for combination in parameter_combinations:
            # Map combination to parameter names
            params = dict(zip(parameter_grid.keys(), combination))
            
            # Extract hyperparameters
            vns_max_duration = params.get('vns_max_duration', 60)
            vns_neighborhood_degree = params.get('vns_neighborhood_degree', 2)
            vns_neighborhood = params.get('vns_neighborhood', n.multi_opt_neighborhood)
            
            print(f"Processing instance {i + 1} with parameters: {params}")

            try:
                # VNS + Hill Climbing
                start_time_vns = time.time()
                vns_solution, vns_profit = vns_hill_climbing(
                    instance['N'],
                    greedy_solution,
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    generate_neighbors=vns_neighborhood,
                    max_time=vns_max_duration,
                    k_max=vns_neighborhood_degree
                )
                vns_time = time.time() - start_time_vns

                # Record results
                results.append({
                    'Instance': i + 1,
                    'vns_max_duration': vns_max_duration,
                    'vns_neighborhood_degree': vns_neighborhood_degree,
                    'vns_neighborhood': vns_neighborhood.__name__,
                    'VNS Profit': vns_profit,
                    'VNS Time': vns_time,
                    'VNS Ecart': abs(instance['optimal_value'] - greedy_profit)
                })
            except Exception as e:
                print(f"Error with parameters {params}: {e}")

    # Save results for this instance file
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{instance_name}.csv")
    results_df.to_csv(output_file, index=False)

    print("\nVNS grid search completed. Results saved to:", output_dir)
    return results_df

def grid_search_sa(data, instance_name, parameter_grid, max_instances=5):
    """
    Performs a grid search over hyperparameters for the Simulated Annealing metaheuristic.

    Args:
        instance_name (str): Name of the instance file to read.
        parameter_grid (dict): Dictionary of SA hyperparameters and their potential values.
        max_instances (int): Maximum number of instances to process.

    Returns:
        pd.DataFrame: A DataFrame with results for all parameter combinations.
    """
    # Generate all combinations of hyperparameters
    parameter_combinations = list(itertools.product(*parameter_grid.values()))

    # Create a DataFrame to store results
    results = []

    # Ensure output directory exists
    output_dir = os.path.join("grid_search", "sa")
    ensure_directory_exists(output_dir)

    # Iterate through instances
    for i, instance in enumerate(data):
        if i >= max_instances:
            break

        print(f"\nProcessing instance {i + 1} for SA...")
        
        # Prepare instance data
        profits = np.array(instance['profits'], dtype=np.float64)
        resource_consumption = np.array(instance['resource_consumption'], dtype=np.float64)
        resource_availabilities = np.array(instance['resource_availabilities'], dtype=np.float64)
        
        # Generate an initial solution using the greedy heuristic
        greedy_solution, greedy_profit = greedy_heuristic(
                instance['N'],
                profits,
                resource_consumption,
                resource_availabilities
            )

        for combination in parameter_combinations:
            # Map combination to parameter names
            params = dict(zip(parameter_grid.keys(), combination))
            
            # Extract hyperparameters
            sa_cooling_rate = params.get('sa_cooling_rate', 0.95)
            sa_neighborhood_degree = params.get('sa_neighborhood_degree', 2)
            sa_neighborhood = params.get('sa_neighborhood', n.multi_opt_neighborhood)
            sa_max_duration = params.get('sa_max_duration', 300)
            sa_iter_max = params.get('sa_max_iter', 1000)
            
            print(f"Processing instance {i + 1} with parameters: {params}")

            try:
                # Simulated Annealing
                start_time_sa = time.time()
                sa_solution, sa_profit = simulated_annealing_metaheuristic(
                    instance['N'],
                    resource_consumption,
                    resource_availabilities,
                    profits,
                    generate_neighbors=sa_neighborhood,
                    initial_solution=greedy_solution,
                    max_time=sa_max_duration,
                    iter_max=sa_iter_max,
                    cooling_rate=sa_cooling_rate,
                    k=sa_neighborhood_degree
                )
                sa_time = time.time() - start_time_sa

                # Record results
                results.append({
                    'Instance': i + 1,
                    'sa_cooling_rate': sa_cooling_rate,
                    'sa_neighborhood_degree': sa_neighborhood_degree,
                    'sa_neighborhood': sa_neighborhood.__name__,
                    'sa_max_iter': sa_iter_max,
                    'sa_max_duration': sa_max_duration,
                    'SA Profit': sa_profit,
                    'SA Time': sa_time,
                    'SA Ecart': abs(instance['optimal_value'] - greedy_profit)
                })
            except Exception as e:
                print(f"Error with parameters {params}: {e}")

    # Save results for this instance
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{instance_name}.csv")
    results_df.to_csv(output_file, index=False)

    print("\nSA grid search completed. Results saved to:", output_dir)
    return results_df


def plot_results(instance_name, method):
    """
    Reads grid search results from files and plots performance graphs.

    Args:
        instance_name (str): Name of the instance file to analyze.
        method (str): Either "vns" or "sa".

    Returns:
        None
    """
    input_dir = os.path.join("grid_search", method)
    instance_files = [f for f in os.listdir(input_dir) if f.startswith(instance_name) and f.endswith(".csv")]

    for file in instance_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        if method == "vns":
            for degree in df['vns_neighborhood_degree'].unique():
                subset = df[df['vns_neighborhood_degree'] == degree]
                plt.plot(subset['vns_max_duration'], subset['VNS Profit'], label=f"Degree {degree}")
            plt.title(f"VNS Results for {file}")
            plt.xlabel("Max Duration")
            plt.ylabel("Profit")
            plt.legend()
            plt.grid()
            plt.show()

        elif method == "sa":
            for cooling_rate in df['sa_cooling_rate'].unique():
                subset = df[df['sa_cooling_rate'] == cooling_rate]
                plt.plot(subset['sa_max_duration'], subset['SA Profit'], label=f"Cooling Rate {cooling_rate}")
            plt.title(f"SA Results for {file}")
            plt.xlabel("Max Duration")
            plt.ylabel("Profit")
            plt.legend()
            plt.grid()
            plt.show()
