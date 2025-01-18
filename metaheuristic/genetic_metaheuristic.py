import random
from deap import base, creator, tools, algorithms
from utilities import calculate_profit, is_feasible
from heuristics.repair_heuristic import repair_heuristic
import numpy as np


def genetic_metaheuristic(N, M, resource_consumption, resource_availabilities, profits):
    """
    Implements a genetic algorithm to solve the multidimensional knapsack problem.

    Args:
        N (int): Number of items.
        M (int): Number of resources.
        resource_consumption (np.ndarray): Resource consumption matrix (M x N).
        resource_availabilities (np.ndarray): Available resources for each type.
        profits (np.ndarray): Array of profits for each item.

    Returns:
        tuple:
            - list: Best solution found (binary vector).
            - float: Total profit of the best solution.
    """
    random.seed(42)  # Set seed for reproducibility

    # Ensure the previous DEAP creator classes are cleared
    if "Fitness" in creator.__dict__:
        del creator.Fitness
    if "Individual" in creator.__dict__:
        del creator.Individual

    # Define the fitness and individual classes
    creator.create("Fitness", base.Fitness, weights=(1.0,))  # Maximization problem
    creator.create("Individual", list, fitness=creator.Fitness)

    # Initialize the DEAP toolbox
    toolbox = base.Toolbox()

    # Generate feasible individuals using the repair heuristic
    def generate_feasible_individual():
        initial_solution = [random.randint(0, 1) for _ in range(N)]
        repaired_solution, _ = repair_heuristic(
            initial_solution, N, M, resource_consumption, resource_availabilities, profits
        )
        return creator.Individual(repaired_solution)
    
    toolbox.register("individual", generate_feasible_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness evaluation function
    def evaluate(individual):
        total_weight = [0] * M
        total_profit = 0

        # Calculate total profit and resource consumption
        for i in range(N):
            if individual[i] == 1:  # If the item is selected
                total_profit += profits[i]
                for d in range(M):
                    total_weight[d] += resource_consumption[d][i]

        # Check resource constraints
        if all(total_weight[d] <= resource_availabilities[d] for d in range(M)):
            return total_profit,
        else:
            # Penalize solutions violating constraints
            penalty = sum(max(0, total_weight[d] - resource_availabilities[d]) for d in range(M))
            return total_profit - penalty,

    # Uniform crossover operator
    def uniform_crossover(parent1, parent2, indpb):
        assert len(parent1) == len(parent2), "Parents must be of the same length"
        offspring1 = parent1[:]
        offspring2 = parent2[:]

        for i in range(len(parent1)):
            if random.random() < indpb:  # Swap genes with probability indpb
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

        return creator.Individual(offspring1), creator.Individual(offspring2)

    # Bit-flip mutation operator
    def bit_flip_mutation(individual, indpb):
        """
        Perform bit-flip mutation on a binary individual.

        Args:
            individual (list): The binary individual to mutate.
            indpb (float): Probability of flipping each bit.

        Returns:
            tuple: Mutated individual.
        """
        for i in range(len(individual)):
            if random.random() < indpb:  # Flip bit with probability indpb
                individual[i] = 1 - individual[i]
        return individual,

    # Register functions in the toolbox
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", uniform_crossover, indpb=0.5)
    toolbox.register("mutate", bit_flip_mutation, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=200)

    # Configure statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=150, stats=stats, verbose=True
    )

    # Extract the best solution
    best_ind = tools.selBest(pop, 1)[0]
    total_profit = calculate_profit(best_ind, profits)
    print("\nBest Solution:", best_ind)
    print("\nTotal Profit:", total_profit)

    # Ensure the solution is feasible
    feasible = is_feasible(
        solution=best_ind, M=M, 
        resource_consumption=resource_consumption, 
        resource_availabilities=resource_availabilities
    )

    while not feasible:
        best_ind, total_profit = repair_heuristic(
            initial_solution=best_ind, N=N, M=M, 
            resource_consumption=resource_consumption, 
            resource_availabilities=resource_availabilities, profits=profits
        )
        feasible = is_feasible(
            solution=best_ind, M=M, 
            resource_consumption=resource_consumption, 
            resource_availabilities=resource_availabilities
        )
    print("\nFeasibility:", feasible)
    
    return best_ind, total_profit


if __name__ == "__main__":
    genetic_metaheuristic()
