import random
import numpy as np
import tsplib95
import matplotlib.pyplot as plt

class HarmonySearch:
    def __init__(self, problem, hms=10, hmcr=0.9, par=0.3, max_iter=600):
        self.problem = problem
        self.num_cities = len(problem.node_coords)  # Nombre de villes dans le problème
        self.hms = hms  # Taille de la mémoire d'harmonie
        self.hmcr = hmcr  # Taux de considération de la mémoire d'harmonie
        self.par = par  # Taux d'ajustement de pitch
        self.max_iter = max_iter  # Nombre maximal d'itérations
        self.harmony_memory = []  # Mémoire d'harmonie
        self.best_routes = []  # Stocker la meilleure route pour chaque itération
        self.best_distances = []  # Stocker la meilleure distance pour chaque itération

    def initialize_harmony_memory(self):
        # Initialiser la mémoire d'harmonie avec des solutions aléatoires
        for _ in range(self.hms):
            solution = list(range(1, self.num_cities + 1))  # TSPLIB utilise une indexation à partir de 1
            random.shuffle(solution)  # Mélanger les villes pour obtenir une solution aléatoire
            self.harmony_memory.append(solution)

    def calculate_distance(self, solution):
        # Calculer la distance totale d'une solution (route)
        return self.problem.trace_tours([solution])[0]

    def create_new_harmony(self):
        # Créer une nouvelle harmonie (solution)
        new_harmony = [-1] * self.num_cities  # Initialiser la nouvelle harmonie avec des valeurs -1
        for i in range(self.num_cities):
            if random.random() < self.hmcr:
                # Choisir depuis la mémoire d'harmonie
                new_harmony[i] = self.harmony_memory[random.randint(0, self.hms-1)][i]
                if random.random() < self.par:
                    # Ajustement de pitch
                    new_harmony[i] = (new_harmony[i] - 1 + random.choice([-1, 1])) % self.num_cities + 1
            else:
                # Sélection aléatoire
                new_harmony[i] = random.randint(1, self.num_cities)
        
        # S'assurer qu'il n'y a pas de doublons dans la nouvelle harmonie
        new_harmony = list(dict.fromkeys(new_harmony))
        missing = set(range(1, self.num_cities + 1)) - set(new_harmony)
        new_harmony.extend(missing)
        return new_harmony

    def update_harmony_memory(self, new_harmony):
        # Mettre à jour la mémoire d'harmonie si la nouvelle harmonie est meilleure que la pire harmonie actuelle
        worst_index = max(range(self.hms), key=lambda i: self.calculate_distance(self.harmony_memory[i]))
        if self.calculate_distance(new_harmony) < self.calculate_distance(self.harmony_memory[worst_index]):
            self.harmony_memory[worst_index] = new_harmony

    def plot_route(self, route, iteration):
        # Tracer la route (solution) actuelle
        coordinates = [self.problem.node_coords[i] for i in route]
        coordinates.append(coordinates[0])  # Retourner au point de départ
        x, y = zip(*coordinates)
        
        plt.plot(x, y, 'bo-')
        plt.title(f'Meilleure route à l\'itération {iteration}')
        plt.xlabel('Coordonnée X')
        plt.ylabel('Coordonnée Y')
        for i, (x, y) in enumerate(coordinates[:-1]):
            plt.annotate(str(route[i]), (x, y), xytext=(5, 5), textcoords='offset points')
        plt.draw()
        plt.pause(0.01)

    def solve(self):
        # Résoudre le problème en utilisant l'algorithme de recherche par harmonie
        self.initialize_harmony_memory()
        
        plt.figure(figsize=(10, 10))
        for iteration in range(self.max_iter):
            new_harmony = self.create_new_harmony()  # Créer une nouvelle harmonie
            self.update_harmony_memory(new_harmony)  # Mettre à jour la mémoire d'harmonie
            
            best_solution = min(self.harmony_memory, key=self.calculate_distance)  # Meilleure solution actuelle
            best_distance = self.calculate_distance(best_solution)  # Distance de la meilleure solution
            
            self.best_routes.append(best_solution)  # Ajouter la meilleure solution à la liste
            self.best_distances.append(best_distance)  # Ajouter la meilleure distance à la liste
            
            plt.clf()  # Effacer la figure actuelle
            self.plot_route(best_solution, iteration)  # Tracer la meilleure route
        
        plt.show()  # Afficher le graphique final
        
        overall_best = min(self.best_routes, key=self.calculate_distance)  # Trouver la meilleure solution globale
        overall_best_distance = self.calculate_distance(overall_best)  # Distance de la meilleure solution globale
        return overall_best, overall_best_distance

# Fonction de Rastrigin pour les benchmarks
def rastrigin(solution):
    A = 10
    return A * len(solution) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in solution])

# Fonction sphère pour les benchmarks
def sphere(solution):
    return sum([x**2 for x in solution])

def run_benchmarks(problem, best_route, benchmark_func, benchmark_name):
    # Exécuter les benchmarks en utilisant la meilleure route trouvée
    x_coords = [problem.node_coords[node][0] for node in best_route]  # Extraire les coordonnées x des nœuds
    performance = benchmark_func(x_coords)  # Calculer la performance en utilisant la fonction de benchmark
    print(f"Performance du benchmark {benchmark_name}: {performance}")

# Exemple d'utilisation
problem = tsplib95.load('dataset.tsp')  # Charger le problème TSPLIB
hs = HarmonySearch(problem)  # Initialiser l'algorithme de recherche par harmonie
best_route, best_distance = hs.solve()  # Résoudre le problème et obtenir la meilleure route et distance

print(f"Meilleure route: {best_route}")  # Afficher la meilleure route
print(f"Meilleure distance: {best_distance}")  # Afficher la meilleure distance

# Évaluer la performance en utilisant la fonction de Rastrigin
run_benchmarks(problem, best_route, rastrigin, "Rastrigin")

# Évaluer la performance en utilisant la fonction sphère
run_benchmarks(problem, best_route, sphere, "Sphère")

# Tracer la convergence de l'algorithme
plt.figure(figsize=(10, 5))
plt.plot(range(hs.max_iter), hs.best_distances)
plt.title('Convergence de la recherche par harmonie')
plt.xlabel('Itération')
plt.ylabel('Meilleure distance')
plt.show()
