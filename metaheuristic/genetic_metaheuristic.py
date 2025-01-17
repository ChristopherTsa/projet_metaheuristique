import random
from deap import base, creator, tools, algorithms
from functools import partial
from utilities import calculate_profit, is_feasible
from heuristics.repair_heuristic import repair_heuristic
import numpy as np



def genetic_metaheuristic(N, M, resource_consumption, resource_availabilities, profits):
    random.seed(42)

    # Création de la classe Fitness et Individu
    if "Fitness" in creator.__dict__:
        del creator.Fitness

    if "Individual" in creator.__dict__:
        del creator.Individual
    

    # Création de la classe Fitness et Individu
    creator.create("Fitness", base.Fitness, weights=(1.0,))  # Maximisation
    creator.create("Individual", list, fitness=creator.Fitness)

    # Toolbox pour l'algorithme génétique
    toolbox = base.Toolbox()

    # Génération des individus et de la population
    def generate_feasible_individual():
        initial_solution = [random.randint(0, 1) for _ in range(N)]
        repaired_solution, _ = repair_heuristic(
            initial_solution, N, M, resource_consumption, resource_availabilities, profits
        )
        return creator.Individual(repaired_solution)
    
    toolbox.register("individual", generate_feasible_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        total_weight = [0] * M
        total_profit = 0

        for i in range(N):
            if individual[i] == 1:  # Si l'objet est sélectionné
                total_profit += profits[i]
                for d in range(M):
                    total_weight[d] += resource_consumption[d][i]

        # Vérification des contraintes
        if all(total_weight[d] <= resource_availabilities[d] for d in range(M)):
            return total_profit,
        else:
            # Pénalisation si les contraintes sont violées
            penalty = sum(max(0, total_weight[d] - resource_availabilities[d]) for d in range(M))
            return total_profit - penalty,
    
    def uniform_crossover(parent1, parent2, indpb):
    
        # Vérification que les deux parents ont la même longueur
        assert len(parent1) == len(parent2), "Parents must be of the same length"

        offspring1 = parent1[:]
        offspring2 = parent2[:]

        # Réalisation du croisement uniforme
        for i in range(len(parent1)):
            if random.random() < indpb:  # Swap des gênes avec la probabilité indpb
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

        return creator.Individual(offspring1), creator.Individual(offspring2)
    
    def bit_flip_mutation(individual, indpb):
        """
        Perform bit-flip mutation on a binary individual.

        Args:
            individual (list): The binary individual to mutate.
            indpb (float): Probability of flipping each bit.

        Returns:
            tuple: A tuple containing the mutated individual.
        """
        for i in range(len(individual)):
            if random.random() < indpb:  # Mutuation du bit avec la probabilité indpb
                individual[i] = 1 - individual[i]  # Flip 0 to 1 or 1 to 0
        return individual,

    toolbox.register("evaluate", evaluate)
    
    toolbox.register("mate", uniform_crossover, indpb=0.5)
    
    toolbox.register("mutate", bit_flip_mutation, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    

    # Initialisation de la population
    pop = toolbox.population(n=200)

    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Exécution de l'algorithme génétique
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=150, \
                                   stats=stats, verbose=True)
    
    # Extraction de la meilleure solution
    best_ind = tools.selBest(pop, 1)[0]
    total_profit = calculate_profit(best_ind, profits)
    print("\nMeilleure solution:", best_ind)
    print("\nProfit total:", total_profit)
    feasible = is_feasible(solution=best_ind, M=M, resource_consumption=resource_consumption, resource_availabilities=resource_availabilities)
    while not feasible:
        best_ind, total_profit = repair_heuristic(initial_solution=best_ind, N=N, M=M, resource_consumption=resource_consumption, resource_availabilities=resource_availabilities, profits=profits)
        feasible = is_feasible(solution=best_ind, M=M, resource_consumption=resource_consumption, resource_availabilities=resource_availabilities)
    print("\nFaisabilité : ", feasible)
    
    return best_ind, total_profit
if __name__ == "__main__":
    genetic_metaheuristic()