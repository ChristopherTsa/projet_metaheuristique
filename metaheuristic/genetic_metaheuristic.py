import random
from deap import base, creator, tools, algorithms
from functools import partial
from utilities import calculate_profit, is_feasible
from heuristics.repair_heuristic import repair_heuristic
import numpy as np




# Algorithme principal
def genetic_metaheuristic(N, M, resource_consumption, resource_availabilities, profits):
    random.seed(42)  # Pour la reproductibilité

    

    # Création de la classe Fitness et Individu
    creator.create("Fitness", base.Fitness, weights=(1.0,))  # Maximisation
    creator.create("Individual", list, fitness=creator.Fitness)

    # Toolbox pour l'algorithme génétique
    toolbox = base.Toolbox()

    # Génération des individus et de la population
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual, N, M, resource_consumption, resource_availabilities, profits):
        try:
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
        except Exception as e:
            return 0,
    print(100)
    # Enregistrement des fonctions dans la toolbox
    #toolbox.register("evaluate", evaluate)
    toolbox.register(
    "evaluate", 
    partial(evaluate, N=N, M=M, resource_consumption=resource_consumption, 
            resource_availabilities=resource_availabilities, profits=profits)
    )
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    print(110)

    # Initialisation de la population
    pop = toolbox.population(n=400)

    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    print(120)
    # Exécution de l'algorithme génétique
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=400, \
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
    """
    print("Profit total:", evaluate(best_ind)[0], N=N, M=M, resource_consumption=resource_consumption, 
            resource_availabilities=resource_availabilities, profits=profits)
    
    print("Poids total par ressource:", [
        sum(resource_consumption[d][i] * best_ind[i] for i in range(N)) for d in range(M)
    ])
    """
    return best_ind, total_profit
if __name__ == "__main__":
    genetic_metaheuristic()