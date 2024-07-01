import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice as np_choice
import random as rn

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Initialisation de l'algorithme de colonie de fourmis.
        """
        self.distances = distances  # Matrice des distances entre les nœuds
        self.pheromone = np.ones(self.distances.shape) / len(distances)  # Matrice des phéromones initialisée
        self.all_inds = range(len(distances))  # Indices de tous les nœuds
        self.n_ants = n_ants  # Nombre de fourmis
        self.n_best = n_best  # Nombre de meilleures fourmis à considérer
        self.n_iterations = n_iterations  # Nombre d'itérations
        self.decay = decay  # Taux de décroissance des phéromones
        self.alpha = alpha  # Importance des phéromones
        self.beta = beta  # Importance des distances

    def run(self):
        """
        Exécution de l'algorithme de colonie de fourmis.
        """
        shortest_path = None  # Chemin le plus court de l'itération courante
        all_time_shortest_path = ("placeholder", np.inf)  # Meilleur chemin trouvé jusqu'à présent
        plt.ion()  # Mode interactif pour la visualisation
        fig, ax = plt.subplots()  # Création de la figure et des axes pour la visualisation
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()  # Génération de tous les chemins pour les fourmis
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)  # Mise à jour des phéromones
            shortest_path = min(all_paths, key=lambda x: x[1])  # Sélection du chemin le plus court de l'itération
            print(f"Iteration {i+1}: {shortest_path}")  # Affichage du chemin le plus court de l'itération courante
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path  # Mise à jour du meilleur chemin trouvé jusqu'à présent
            self.pheromone = self.pheromone * self.decay  # Application de la décroissance des phéromones
            
            # Visualisation du chemin le plus court de l'itération courante
            self.plot(shortest_path[0], f"Iteration {i+1}: Distance {shortest_path[1]:.2f}", fig, ax)
            
        plt.ioff()  # Désactivation du mode interactif
        plt.show()  # Affichage de la visualisation finale
        return all_time_shortest_path  # Retour du meilleur chemin trouvé

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        """
        Mise à jour des phéromones basée sur les meilleurs chemins.
        """
        sorted_paths = sorted(all_paths, key=lambda x: x[1])  # Tri des chemins par distance
        for path, dist in sorted_paths[:n_best]:  # Considération des n meilleurs chemins
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]  # Mise à jour des phéromones

    def gen_path_dist(self, path):
        """
        Calcul de la distance totale d'un chemin.
        """
        total_dist = 0  # Initialisation de la distance totale
        for ele in path:
            total_dist += self.distances[ele]  # Calcul de la distance totale pour un chemin donné
        return total_dist  # Retour de la distance totale

    def gen_all_paths(self):
        """
        Génération de tous les chemins pour les fourmis.
        """
        all_paths = []  # Initialisation de la liste de tous les chemins
        for i in range(self.n_ants):
            path = self.gen_path(0)  # Génération d'un chemin à partir du nœud de départ 0
            all_paths.append((path, self.gen_path_dist(path)))  # Ajout du chemin et de sa distance à la liste
        return all_paths  # Retour de la liste de tous les chemins

    def gen_path(self, start):
        """
        Génération d'un chemin pour une fourmi à partir d'un nœud de départ.
        """
        path = []  # Initialisation du chemin
        visited = set()  # Initialisation de l'ensemble des nœuds visités
        visited.add(start)  # Ajout du nœud de départ aux nœuds visités
        prev = start  # Initialisation du nœud précédent
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)  # Choix du prochain nœud
            path.append((prev, move))  # Ajout du mouvement au chemin
            prev = move  # Mise à jour du nœud précédent
            visited.add(move)  # Ajout du nœud visité à l'ensemble des nœuds visités
        path.append((prev, start))  # Retour au nœud de départ
        return path  # Retour du chemin généré

    def pick_move(self, pheromone, dist, visited):
        """
        Choix du prochain nœud à visiter par une fourmi.
        """
        pheromone = np.copy(pheromone)  # Copie des phéromones pour éviter les modifications directes
        pheromone[list(visited)] = 0  # Mise à zéro des phéromones des nœuds déjà visités

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)  # Calcul de l'influence des phéromones et des distances

        norm_row = row / row.sum()  # Normalisation des probabilités de mouvement
        move = np_choice(self.all_inds, 1, p=norm_row)[0]  # Choix du prochain nœud basé sur les probabilités
        return move  # Retour du nœud choisi

    def plot(self, path, title, fig, ax):
        """
        Visualisation du chemin courant.
        """
        ax.clear()  # Effacement de l'axe précédent
        nodes = range(len(self.distances))  # Liste des nœuds
        coords = {i: (rn.uniform(0, 10), rn.uniform(0, 10)) for i in nodes}  # Génération de coordonnées aléatoires pour les nœuds
        for key, value in coords.items():
            ax.scatter(*value)  # Tracé des nœuds
            ax.annotate(key, value)  # Annotation des nœuds
        for move in path:
            start, end = move
            ax.plot([coords[start][0], coords[end][0]], [coords[start][1], coords[end][1]], 'b')  # Tracé des mouvements entre les nœuds
        ax.set_title(title)  # Titre de l'axe
        plt.draw()  # Redessin de la figure
        plt.pause(0.01)  # Pause pour l'affichage interactif
