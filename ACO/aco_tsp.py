import numpy as np
from aco import AntColony
import tsplib95

def load_tsplib_data(file_path):
    problem = tsplib95.load(file_path)
    nodes = list(problem.get_nodes())
    distances = np.zeros((len(nodes), len(nodes)))
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
            else:
                distances[i, j] = np.inf
    return distances

# Rastrigin function
def rastrigin(solution):
    A = 10
    return A * len(solution) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in solution])

# Sphere function
def sphere(solution):
    return sum([x**2 for x in solution])

# Function to run benchmarks
def run_benchmarks(shortest_path, benchmark_func, benchmark_name):
    #all_time_shortest_path = ant_colony.run()
    solution = [move[0] for move in shortest_path[0]]  # Convert path to solution
    performance = benchmark_func(solution)
    print(f"{benchmark_name} Benchmark Performance: {performance}")

# Example usage
file_path = 'ACO/xqf131.tsp'
distances = load_tsplib_data(file_path)

# Initialize Ant Colony Optimization with TSPLIB distances
ant_colony = AntColony(distances, 100, 20, 5, 0.95, alpha=1, beta=2)

# Run the algorithm
shortest_path = ant_colony.run()
print(f"shortest_path: {shortest_path[0]}")
print(f"Distance: {shortest_path[1]}")


# Benchmark the performance using Rastrigin function
run_benchmarks(shortest_path, rastrigin, "Rastrigin")

# Benchmark the performance using Sphere function
run_benchmarks(shortest_path, sphere, "Sphere")
