# Metaheuristic-algorithms-ACO-AFSA-BHA
METAHEURISTICS AND STOCHASTIC SEARCH ALGORITHMS - ACO - AFSA - BHA

## **Algorithms**

- [x] Ant colony optimization - ACO
- [ ] Artificial Fish Swarm Algorithm - AFSA
- [ ] Black Holes Algorithm - BHA

## **Algorithm Details**
### **Ant colony optimization - ACO**


**Algorithm Steps**



**Parameters**

- `noThieves`: The number of thieves in the population.

- `distances (2D numpy.array)`: Square matrix of distances. Diagonal is assumed to be np.inf.
- `n_ants (int)`: Number of ants running per iteration
- `n_best (int)`: Number of best ants who deposit pheromone
- `n_iteration (int)`: Number of iterations
- `decay (float)`: Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
- `alpha (int or float)`: exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
- `beta (int or float)`: exponent on distance, higher beta give distance more weight. Default=1