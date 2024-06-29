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

# Example usage
file_path = 'ACO/xqf131.tsp'
distances = load_tsplib_data(file_path)

# Initialize Ant Colony Optimization with TSPLIB distances
ant_colony = AntColony(distances, 100, 20, 1000, 0.95, alpha=1, beta=2)

# Run the algorithm
shortest_path = ant_colony.run()
print(f"shortest_path: {shortest_path}")
# Plot the result
ant_colony.plot(shortest_path[0])

# Benchmark the performance using CEC 2016 and CEC 2020
# Implement and call your benchmark functions here