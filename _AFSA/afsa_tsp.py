import random
import tsplib95
import matplotlib.pyplot as plt
import time
import statistics

# Classe représentant un poisson artificiel
class ArtificialFish:
    def __init__(self, tour):
        self.tour = tour
        self.fitness = float('inf')

    # Calcul de la fitness du poisson
    def calculate_fitness(self, problem):
        self.fitness = sum(problem.get_weight(self.tour[i], self.tour[i+1]) for i in range(len(self.tour)-1))
        self.fitness += problem.get_weight(self.tour[-1], self.tour[0])
        return self.fitness

# Initialisation de la population de poissons
def initialize_population(problem, pop_size):
    cities = list(problem.get_nodes())
    return [ArtificialFish(random.sample(cities, len(cities))) for _ in range(pop_size)]

# Vérification si un voisin est dans le champ de vision du poisson
def visual(fish, neighbor, visual_scope):
    return sum(1 for i in range(len(fish.tour)) if fish.tour[i] != neighbor.tour[i]) <= visual_scope

# Trouver le meilleur voisin d'un poisson
def find_best_neighbor(fish, population, problem, visual_scope):
    neighbors = [f for f in population if visual(fish, f, visual_scope)]
    if not neighbors:
        return None
    return min(neighbors, key=lambda x: x.fitness)

# Suivre le meilleur voisin
def follow(fish, best_neighbor, problem):
    i, j = random.sample(range(len(fish.tour)), 2)
    new_tour = fish.tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    new_fish = ArtificialFish(new_tour)
    new_fish.calculate_fitness(problem)
    return new_fish if new_fish.fitness < fish.fitness else fish

# Mouvements de banc de poissons
def swarm(fish, population, problem, visual_scope):
    center = [0] * len(fish.tour)
    count = 0
    for f in population:
        if visual(fish, f, visual_scope):
            for i in range(len(fish.tour)):
                center[i] += f.tour[i]
            count += 1
    if count > 0:
        center = [c // count for c in center]
        new_tour = fish.tour[:]
        for i in range(len(fish.tour)):
            j = new_tour.index(center[i])
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_fish = ArtificialFish(new_tour)
        new_fish.calculate_fitness(problem)
        return new_fish if new_fish.fitness < fish.fitness else fish
    return fish

# Mouvement de chasse
def prey(fish, problem):
    i, j = random.sample(range(len(fish.tour)), 2)
    new_tour = fish.tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    new_fish = ArtificialFish(new_tour)
    new_fish.calculate_fitness(problem)
    return new_fish if new_fish.fitness < fish.fitness else fish

# Algorithme de banc de poissons artificiel
def artificial_fish_swarm(problem, pop_size=50, max_iterations=1000, visual_scope=5):
    population = initialize_population(problem, pop_size)
    for fish in population:
        fish.calculate_fitness(problem)
    
    best_fish = min(population, key=lambda x: x.fitness)
    
    # Initialisation du tracé
    fig, ax = plt.subplots()
    ax.set_title('Meilleure Fitness au fil des itérations')
    ax.set_xlabel('Itération')
    ax.set_ylabel('Fitness')
    iterations = []
    best_fitnesses = []

    for iteration in range(max_iterations):
        for i, fish in enumerate(population):
            best_neighbor = find_best_neighbor(fish, population, problem, visual_scope)
            if best_neighbor and best_neighbor.fitness < fish.fitness:
                population[i] = follow(fish, best_neighbor, problem)
            elif random.random() < 0.5:
                population[i] = swarm(fish, population, problem, visual_scope)
            else:
                population[i] = prey(fish, problem)
        
        current_best = min(population, key=lambda x: x.fitness)
        if current_best.fitness < best_fish.fitness:
            best_fish = current_best
        
        # Mise à jour des données du tracé
        iterations.append(iteration)
        best_fitnesses.append(best_fish.fitness)
        
        if iteration % 10 == 0:  # Afficher la progression toutes les 10 itérations
            print(f"Itération {iteration}: Meilleure fitness = {best_fish.fitness}")
        
        # Mise à jour du tracé
        ax.plot(iterations, best_fitnesses, color='b', marker='o')
        fig.canvas.draw()
        plt.pause(0.001)

    plt.show()
    
    return best_fish

# Benchmark de temps d'exécution
def time_benchmark(problem, num_runs=5, pop_size=50, max_iterations=100, visual_scope=5):
    execution_times = []
    optimal_solutions = []

    for run in range(num_runs):
        start_time = time.time()
        best_solution = artificial_fish_swarm(problem, pop_size, max_iterations, visual_scope)
        end_time = time.time()
        
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        print(f"Run {run+1}: Temps = {execution_time:.4f} secondes, Fitness = {best_solution.fitness}")
        optimal_solutions.append(best_solution.fitness)
    
    avg_time = statistics.mean(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)
    std_dev_time = statistics.stdev(execution_times)
    optimal_solution = min(optimal_solutions)

    print("\nRésultats du Benchmark de Temps:")
    print(f"Temps d'exécution moyen: {avg_time:.4f} secondes")
    print(f"Temps d'exécution minimum: {min_time:.4f} secondes")
    print(f"Temps d'exécution maximum: {max_time:.4f} secondes")
    print(f"Écart-type du temps d'exécution: {std_dev_time:.4f} secondes")
    print(f"Solution optimale: {optimal_solution}")

# Exemple d'utilisation
problem = tsplib95.load('dataset.tsp')
time_benchmark(problem)
