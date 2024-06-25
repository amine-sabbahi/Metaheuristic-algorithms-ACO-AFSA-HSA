import matplotlib.pyplot as plt
import random, math
from scipy.spatial import distance
from operator import attrgetter
import copy

# Define the Fish class
class Fish():
    """docstring for Fish"""
    def __init__(self, p, f):
        self.position = p
        self.fitness = f

# Initialize the fish population with random positions
def initialize(dim, pop, GroupFish):
    for i in range(pop):
        s = []
        for j in range(dim):
            s.append(random.random())
        fit = calcFitness(s)
        F = Fish(s, fit)
        GroupFish.append(F)

# Calculate the fitness of a fish based on its position using the Rosenbrock function
def calcFitness(s):
    sum = 0
    i = 0
    while i < (len(s) - 1):
        r = 100 * math.pow((s[i+1] - math.pow(s[i], 2)), 2) + math.pow((s[i] - 1), 2)
        sum = sum + r
        i = i + 1
    return sum

# Generate a temporary new position for a fish
def makeTemp(ind, Visual):
    temp_pos = []
    for i in range(len(ind.position)):
        temp = ind.position[i] + (Visual * random.random())
        temp_pos.append(temp)
    temp_fit = calcFitness(temp_pos)
    F = Fish(temp_pos, temp_fit)
    return F

# Implement the prey behavior
def prey(ind, TF, B, dim, step, n, Visual, GroupFish, j):
    crowdFactor = random.uniform(0.5, 1)
    new_State = []
    for i in range(dim):
        m = ind.position[i] + (((TF.position[i] - ind.position[i]) / distance.euclidean(TF.position, ind.position)) * step * random.random())
        new_State.append(m)
    new_fit = calcFitness(new_State)

    nf = Visual * n

    for i in range(int(round(nf))):
        Xc = []
        if j != 0 and j != n - 1:
            print(j)
            for x in range(dim):
                element = GroupFish[j - 1].position[x] + GroupFish[j].position[x] + GroupFish[j + 1].position[x]
                Xc.append(element)

    Xcfitness = calcFitness(Xc)
    
    if B.fitness > ind.fitness and (nf / n) < crowdFactor:
        follow(ind, B, dim, step)
    elif Xcfitness > ind.fitness and (nf / n) < crowdFactor:
        swarm(ind, TF, dim, step)
    else:
        ind.position = new_State
        ind.fitness = new_fit

# Move a fish to a new random position within its visual range
def moveRandomly(ind, Visual):
    new_State = []
    for i in range(len(ind.position)):
        n = ind.position[i] + (Visual * random.random())
        new_State.append(n)
    ind.position = new_State
    ind.fitness = calcFitness(ind.position)

# Get the fish with the best fitness in the group
def getBestFish(GroupFish):
    L = sorted(GroupFish, key=attrgetter('fitness'))
    return L[-1]

# Implement the swarm behavior
def swarm(ind, B, dim, step):
    new_State2 = []
    for i in range(dim):
        m = ind.position[i] + (((B.position[i] - ind.position[i]) / distance.euclidean(B.position, ind.position)) * step * random.random())
        new_State2.append(m)
    ind.position = new_State2
    ind.fitness = calcFitness(new_State2)

# Implement the follow behavior
def follow(ind, TF, dim, step):
    new_State3 = []
    for i in range(dim):
        O = ind.position[i] + (((TF.position[i] - ind.position[i]) / distance.euclidean(TF.position, ind.position)) * step * random.random())
        new_State3.append(O)
    ind.position = new_State3
    ind.fitness = calcFitness(new_State3)

# Main code to run the algorithm
dim = 2  # Dimension of the problem
population = 10  # Population size
GroupFish = []  # List to store the fish population
trytimes = 3  # Number of attempts for each fish to find a better position
Visual = 0.2  # Visual range of the fish
step = 0.3  # Step size for the fish movement
iteration = 10  # Number of iterations
StoreBest = []  # List to store the best fish found in each iteration

# Initialize the fish population
initialize(dim, population, GroupFish)
# Get the best fish in the initial population
B = getBestFish(GroupFish)
StoreBest.append(copy.deepcopy(B))

# Run the main iteration loop
i = 0
while i < iteration:
    j = 0
    while j < population:
        k = 0
        while k < trytimes:
            temp_Position = makeTemp(GroupFish[j], Visual)
            if GroupFish[j].fitness < temp_Position.fitness:
                prey(GroupFish[j], temp_Position, B, dim, step, population, Visual, GroupFish, j)
                break
            k += 1
        moveRandomly(GroupFish[j], Visual)
        j += 1
    i += 1
    B = getBestFish(GroupFish)
    print(B)
    StoreBest.append(copy.deepcopy(B))

# Get the best fish found during the entire process
BE = getBestFish(StoreBest)
print("Best fitness found:", BE.fitness)

# Visualization of the fitness over iterations
fitness_values = [fish.fitness for fish in StoreBest]
plt.plot(range(iteration + 1), fitness_values, marker='o')
plt.title("Fitness Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()
