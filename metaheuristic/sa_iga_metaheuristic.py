import random
from deap import base, creator, tools
import numpy as np
from utilities import calculate_profit, is_feasible
from simulated_annealing_metaheuristic import simulated_annealing_metaheuristic


def sa_iga_metaheuristic(N, M, resource_consumption, resource_availabilities, profits, generate_neighbors, 
                         initial_temperature=None, cooling_rate=0.95, max_iterations_sa=100, epsilon=1e-3, 
                         population_size=400, ngen=200, cxpb=0.7, mutpb=0.3):
    """
    Combines Simulated Annealing (SA) and Genetic Algorithm (GA) to solve the 0/1 MKP.

    Args:
        N (int): Number of projects.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each project.
        generate_neighbors (function): Function to generate a neighbor solution.
        initial_temperature (float): Initial temperature for simulated annealing. If None, it is calculated dynamically.
        cooling_rate (float): Rate at which the temperature decreases.
        max_iterations_sa (int): Maximum iterations for SA.
        epsilon (float): Minimum temperature threshold for SA.
        population_size (int): Size of the population for GA.
        ngen (int): Number of generations for GA.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.

    Returns:
        np.ndarray: Best solution found.
        float: Total profit of the best solution.
    """
    random.seed(42)  # For reproducibility

    # Create DEAP classes
    creator.create("Fitness", base.Fitness, weights=(1.0,))  # Maximization
    creator.create("Individual", list, fitness=creator.Fitness)

    # Toolbox setup
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        """Evaluates an individual based on feasibility and profit."""
        if is_feasible(individual, M, resource_consumption, resource_availabilities):
            return calculate_profit(individual, profits),
        else:
            # Penalize infeasible solutions
            total_weight = [sum(resource_consumption[d][i] * individual[i] for i in range(N)) for d in range(M)]
            penalty = sum(max(0, total_weight[d] - resource_availabilities[d]) for d in range(M))
            return calculate_profit(individual, profits) - penalty,

    # Register functions to the toolbox
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Genetic Algorithm loop
    for gen in range(ngen):
        # Evaluate individuals
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        # Select the next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Apply Simulated Annealing on each individual
        for ind in offspring:
            ind_array = np.array(ind)
            sa_solution, sa_profit = simulated_annealing_metaheuristic(
                N, M, resource_consumption, resource_availabilities, profits, generate_neighbors,
                initial_solution=ind_array, initial_temperature=initial_temperature, 
                cooling_rate=cooling_rate, max_iterations=max_iterations_sa, epsilon=epsilon
            )
            ind[:] = sa_solution.tolist()
            ind.fitness.values = (sa_profit,)

        # Replace population with offspring
        pop[:] = offspring

    # Extract the best individual
    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind), calculate_profit(best_ind, profits)