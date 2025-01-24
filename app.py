import os
import time
import numpy as np
import pandas as pd

import test
import grid_search.grid_search as gs

from utilities import read_knapsack_data
from heuristics.greedy_heuristic import greedy_heuristic
from heuristics.repair_heuristic import repair_heuristic, surrogate_relaxation_mkp

import hill_climbing.neighborhoods as neighborhoods
from hill_climbing.hill_climbing import hill_climbing
from hill_climbing.vns import vns_hill_climbing

from metaheuristic.simulated_annealing_metaheuristic import simulated_annealing_metaheuristic
from metaheuristic.genetic_metaheuristic import genetic_metaheuristic
from metaheuristic.sa_iga_metaheuristic import sa_iga_metaheuristic


if __name__ == "__main__":
    #test.compare_methods("mknap1", max_instances=10)
    #test.compare_methods("mknapcb1", max_instances=10)
    #test.compare_methods("mknapcb2", max_instances=10)
    #test.compare_methods("mknapcb3", max_instances=10)
    #test.compare_methods("mknapcb4", max_instances=10)
    #test.compare_methods("mknapcb5", max_instances=10)
    #test.compare_methods("mknapcb6", max_instances=10)
    #test.compare_methods("mknapcb7", max_instances=10)
    #test.compare_methods("mknapcb8", max_instances=10)
    #test.compare_methods("mknapcb9", max_instances=10)
    
    hc_parameter_grid = {
        'hc_neighbors': [neighborhoods.multi_opt_neighborhood,
                         neighborhoods.multi_swap_neighborhood_indices,
                         neighborhoods.resource_profit_based_reverse_k_neighborhood],
        'hc_k': [1, 2, 3, 4, 5]
    }
    
    vns_parameter_grid = {
        'vns_max_duration': [30, 100, 300],
        'vns_neighborhood_degree': [1, 2],
        'vns_neighborhood_type': [neighborhoods.multi_opt_neighborhood,
                                  neighborhoods.multi_swap_neighborhood_indices,
                                  neighborhoods.resource_profit_based_reverse_k_neighborhood]
    }
    
    sa_parameter_grid = {
        'sa_max_duration': [60, 300, 600],
        'sa_cooling_rate': [0.85, 0.9, 0.95],
        'sa_neighborhood_degree': [1, 2],
        'sa_iter_max': [30, 100, 300],
        'sa_neighborhood_type': [neighborhoods.multi_opt_neighborhood,
                                    neighborhoods.multi_swap_neighborhood_indices,
                                    neighborhoods.resource_profit_based_reverse_k_neighborhood]
    }
    
    ga_parameter_grid = {
        'ga_popsize': [100, 200, 300],
        'ga_cxpb': [0.6, 0.7, 0.8],
        'ga_mutpb': [0.1, 0.2, 0.3],
        'ga_ngen': [100, 150, 200],
        'ga_uniform_crossover_prob': [0.3, 0.5, 0.7],
        'ga_bitflip_mutation_prob': [0.1, 0.2, 0.3],
        'ga_tournament_size': [2, 3, 4]
    }
    
    instance_name = "mknap1"
    max_instances = 10
    data = read_knapsack_data(instance_name)
    print("Instance: ", instance_name)
    
    print("Hill Climbing")
    #gs.grid_search_hc(data, instance_name, hc_parameter_grid, max_instances)
    
    print("VNS")
    gs.grid_search_vns(data, instance_name, vns_parameter_grid, max_instances)
    
    print("Simulated Annealing")
    gs.grid_search_sa(data, instance_name, sa_parameter_grid, max_instances)
    
    print("Genetic Algorithm")
    gs.grid_search_ga(data, instance_name, ga_parameter_grid, max_instances)
    

    #plot_results("mknap1", "vns")
    #plot_results("mknap1", "sa")