import random
import numpy as np
import tsplib95
import matplotlib.pyplot as plt

class HarmonySearch:
    def __init__(self, problem, hms=10, hmcr=0.9, par=0.3, max_iter=1000):
        self.problem = problem
        self.num_cities = len(problem.node_coords)
        self.hms = hms  # Harmony Memory Size
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par  # Pitch Adjustment Rate
        self.max_iter = max_iter
        self.harmony_memory = []
        self.best_routes = []  # Store best route for each iteration
        self.best_distances = []  # Store best distance for each iteration

    def initialize_harmony_memory(self):
        for _ in range(self.hms):
            solution = list(range(1, self.num_cities + 1))  # TSPLIB uses 1-based indexing
            random.shuffle(solution)
            self.harmony_memory.append(solution)

    def calculate_distance(self, solution):
        return self.problem.trace_tours([solution])[0]

    def create_new_harmony(self):
        new_harmony = [-1] * self.num_cities
        for i in range(self.num_cities):
            if random.random() < self.hmcr:
                # Choose from harmony memory
                new_harmony[i] = self.harmony_memory[random.randint(0, self.hms-1)][i]
                if random.random() < self.par:
                    # Pitch adjustment
                    new_harmony[i] = (new_harmony[i] - 1 + random.choice([-1, 1])) % self.num_cities + 1
            else:
                # Random selection
                new_harmony[i] = random.randint(1, self.num_cities)
        
        # Ensure no duplicates
        new_harmony = list(dict.fromkeys(new_harmony))
        missing = set(range(1, self.num_cities + 1)) - set(new_harmony)
        new_harmony.extend(missing)
        return new_harmony

    def update_harmony_memory(self, new_harmony):
        worst_index = max(range(self.hms), key=lambda i: self.calculate_distance(self.harmony_memory[i]))
        if self.calculate_distance(new_harmony) < self.calculate_distance(self.harmony_memory[worst_index]):
            self.harmony_memory[worst_index] = new_harmony

    def plot_route(self, route, iteration):
        coordinates = [self.problem.node_coords[i] for i in route]
        coordinates.append(coordinates[0])  # Return to start
        x, y = zip(*coordinates)
        
        plt.plot(x, y, 'bo-')
        plt.title(f'Best Route at Iteration {iteration}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        for i, (x, y) in enumerate(coordinates[:-1]):
            plt.annotate(str(route[i]), (x, y), xytext=(5, 5), textcoords='offset points')
        plt.draw()
        plt.pause(0.01)

    def solve(self):
        self.initialize_harmony_memory()
        
        plt.figure(figsize=(10, 10))
        for iteration in range(self.max_iter):
            new_harmony = self.create_new_harmony()
            self.update_harmony_memory(new_harmony)
            
            best_solution = min(self.harmony_memory, key=self.calculate_distance)
            best_distance = self.calculate_distance(best_solution)
            
            self.best_routes.append(best_solution)
            self.best_distances.append(best_distance)
            
            plt.clf()  # Clear the current figure
            self.plot_route(best_solution, iteration)
        
        plt.show()  # Show the final plot
        
        overall_best = min(self.best_routes, key=self.calculate_distance)
        overall_best_distance = self.calculate_distance(overall_best)
        return overall_best, overall_best_distance

# Rastrigin function
def rastrigin(solution):
    A = 10
    return A * len(solution) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in solution])

# Sphere function
def sphere(solution):
    return sum([x**2 for x in solution])

def run_benchmarks(problem, best_route, benchmark_func, benchmark_name):
    # Extract x-coordinates of the nodes in the best route
    x_coords = [problem.node_coords[node][0] for node in best_route]
    performance = benchmark_func(x_coords)
    print(f"{benchmark_name} Benchmark Performance: {performance}")


# Example usage
problem = tsplib95.load('dataset.tsp')
hs = HarmonySearch(problem)
best_route, best_distance = hs.solve()

print(f"Best route: {best_route}")
print(f"Best distance: {best_distance}")


# Benchmark the performance using Rastrigin function
run_benchmarks(problem, best_route, rastrigin, "Rastrigin")

# Benchmark the performance using Sphere function
run_benchmarks(problem, best_route, sphere, "Sphere")


# Plot the convergence
plt.figure(figsize=(10, 5))
plt.plot(range(hs.max_iter), hs.best_distances)
plt.title('Convergence of Harmony Search')
plt.xlabel('Iteration')
plt.ylabel('Best Distance')
plt.show()

