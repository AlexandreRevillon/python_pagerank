#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:05:07 2023

@author: arevillo
"""
###################### Import des libraries ######################

import numpy as np
import scipy.linalg as nla
import igraph as ig
import pandas as pd
import csv
import time





#################{# Déclarations des fonctions ###################

def norm(x):
    """Fonction qui calcule la norme euclidienne d'un vecteur

    Args:
        x (vecteur np.array): Vecteur dont on veut calculer la norme

    Returns:
        Réel: Norme euclidienne du vecteur
    """
    somme = 0
    for i in range(x.shape[0]):
        somme += x[i]**2
    return np.sqrt(somme)



def import_tsv():
    """Fonction qui importe le fichier tsv et le transforme en liste de listes

    Returns:
        Liste: Liste de listes contenant les données du fichier tsv
    """
    data = list()
    with open("wikispeedia_paths-and-graph/paths_finished_2.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            try:
                tmp = list()
                tmp.append(row[0])
                tmp.append(int(row[1]))
                tmp.append(int(row[2]))
                tmp.append(row[3].split(";"))
                tmp.append(row[4])
                data.append(tmp)
            except IndexError:
                tmp = list()
    return data



def points_et_transitions():
    """Fonction qui renvoie la liste des points et la liste des transitions

    Returns:
        Liste: Liste des points
        Liste: Liste des transitions
    """
    data = import_tsv()
    points = set()
    transitions = set()
    for ligne in data:
        for i in range(len(ligne[3])):
            if ligne[3][i] != "<":
                points.add(ligne[3][i])
                
        liste_trans = ligne[3]
        if len(liste_trans) > 1:
            i = 0
            while i < len(liste_trans)-1:
                if liste_trans[i+1] == "<":
                    transitions.add((liste_trans[i], liste_trans[i-1]))
                    del liste_trans[i]
                    del liste_trans[i]
                    i -= 1             
                else:
                    transitions.add((liste_trans[i], liste_trans[i+1]))
                    i += 1
    return list(points), list(transitions)



def adjacence_graph():
    """Fonction qui renvoie la matrice d'adjacence du graphe

    Returns:
        DataFrame: Matrice d'adjacence du graphe
    """
    points, transitions = points_et_transitions()
    G = ig.Graph(directed=True)
    for p in points:
        G.add_vertex(name=p)
    G.add_edges(transitions)
    return pd.DataFrame(G.get_adjacency().data, columns=points, index=points)   
    


def pagerank(matrice_adjacence, d=0.85, max_iter=100, tol=1e-6):
    """Fonction permettant de calculer les valeurs pagerank d'une matrice d'adjacence

    Args:
        matrice_adjacence (matrix): matrice d'ajacence du graphe
        d (float, optional): dumping factor. Defaults to 0.85.
        max_iter (int, optional): nombre maximal d'itération. Defaults to 100.
        tol (float, optional): tolérance. Defaults to 1e-6.

    Returns:
        vector: matrice des valeurs de page rank
    """
    n = matrice_adjacence.shape[0]
    
    # Calculer la matrice de transition (stochastique)
    nb_lien_out = np.sum(matrice_adjacence, axis=1)
    matrice_transition = np.zeros((n, n))
    
    for i in range(n):
        if nb_lien_out[i] > 0:
            matrice_transition[i, :] = matrice_adjacence[i, :] / nb_lien_out[i]
        else:
            matrice_transition[i, :] = 1 / n
    
    # Appliquer le facteur d'amortissement d (dumping factor)
    matrice_transition = d * matrice_transition + (1 - d) * (1 / n) * np.ones((n, n))
    
    # Initialiser le vecteur PageRank
    pagerank = np.ones(n) / n
    
    # Appliquer la méthode de la puissance
    for _ in range(max_iter):
        pagerank_new = np.dot(matrice_transition.T, pagerank)
        
        if norm(pagerank_new - pagerank) < tol:
            break
        
        pagerank = pagerank_new
    
    return pagerank
    
    
    
def personalized_pagerank(matrice_adjacence, v, d=0.85, max_iter=100, tol=1e-6):
    """Fonction permettant de calculer les valeurs pagerank d'une matrice d'adjacence à partir d'un vecteur de personalisation

    Args:
        matrice_adjacence (matrix): matrice d'ajacence du graphe
        v (vector): vecteur de personalisation
        d (float, optional): dumping factor. Defaults to 0.85.
        max_iter (int, optional): nombre maximal d'itération. Defaults to 100.
        tol (float, optional): tolérance. Defaults to 1e-6.

    Returns:
        vector: matrice des valeurs de page rank personnalisé
    """
    n = matrice_adjacence.shape[0]
    
    # Vérifier si la somme de v est égale à 1
    assert np.isclose(np.sum(v), 1), "Le vecteur de préférence personnel doit sommer à 1"
    
    # Calculer la matrice de transition (stochastique)
    nb_lien_out = np.sum(matrice_adjacence, axis=1)
    matrice_transition = np.zeros((n, n))
    
    for i in range(n):
        if nb_lien_out[i] > 0:
            matrice_transition[i, :] = matrice_adjacence[i, :] / nb_lien_out[i]
        else:
            matrice_transition[i, :] = 1 / n
    
    # Appliquer le facteur d'amortissement d et le vecteur de préférence personnel v
    matrice_transition = d * matrice_transition + (1 - d) * np.outer(np.ones(n), v)
    
    # Initialiser le vecteur PageRank
    pagerank = np.ones(n) / n
    
    # Appliquer la méthode de la puissance
    for _ in range(max_iter):
        pagerank_new = np.dot(matrice_transition.T, pagerank)
        
        if norm(pagerank_new - pagerank) < tol:
            break
        
        pagerank = pagerank_new
    
    return pagerank
    






###################### Programme principal #######################

# Récupération de la matrice d'adjacence du graphe
A = adjacence_graph()
colnames = A.columns

# Calcul des valeurs de page rank
pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64')), index = colnames)

print("Valeurs de page rank:")
print(pr.sort_values(by=0, ascending=False)*100)
print("\nSomme des valeurs de pagerank: ", pr.sum()[0])

print("\n"*2)

# Variation de paramètre
start = time.time()
pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64'),d=0.15, tol=0.01), index = colnames)
stop = time.time()
duree = stop - start
print("Valeurs de page rank avec d=0.15 et tol=0.01:")
print(pr.sort_values(by=0, ascending=False)*100)
print("Temps d'exécution: ", duree, "s")

start = time.time()
pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64'),d=0.15, tol=0.0001), index = colnames)
stop = time.time()
duree = stop - start
print("\nValeurs de page rank avec d=0.15 et tol=0.0001:")
print(pr.sort_values(by=0, ascending=False)*100)
print("Temps d'exécution: ", duree, "s")

start = time.time()
pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64'),d=0.85, tol=0.01), index = colnames)
stop = time.time()
duree = stop - start
print("\nValeurs de page rank avec d=0.85 et tol=0.01:")
print(pr.sort_values(by=0, ascending=False)*100)
print("Temps d'exécution: ", duree, "s")

start = time.time()
pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64'),d=0.85, tol=0.0001), index = colnames)
stop = time.time()
duree = stop - start
print("\nValeurs de page rank avec d=0.85 et tol=0.0001:")
print(pr.sort_values(by=0, ascending=False)*100)
print("Temps d'exécution: ", duree, "s")





################# Exercice du travail personnel ##################

# #Matrice d'adjacence de l'exercice 1
# A_ex1 = np.array([[0, 1, 1, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0],
#                   [1, 1, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 1, 1],
#                   [0, 0, 0, 1, 0, 1],
#                   [0, 0, 0, 1, 0, 0]], dtype='float64')


# print("\n"*2)

# #Calcul des valeurs de pagerank
# pr=pagerank(A_ex1, 0.85)
# print("\nValeurs de page rank:\n\t", pr)
# print("\tSomme des valeurs de pagerank:", pr.sum())

# #Calcul des valeurs de pagerank personnalisé pour le noeud 1
# personalization = np.zeros(A_ex1.shape[0])
# personalization[0] = 1
# pr=personalized_pagerank(A_ex1, personalization, 0.85)
# print("\nValeurs de page rank avec personnalisation noeud 1:\n\t", pr)
# print("\tSomme des valeurs de pagerank:", pr.sum())


# #Calcul des valeurs de pagerank personnalisé pour le noeud 5
# personalization = np.zeros(A_ex1.shape[0])
# personalization[4] = 1
# pr=personalized_pagerank(A_ex1, personalization, 0.85)
# print("\nValeurs de page rank avec personnalisation noeud 5:\n\t", pr)
# print("\tSomme des valeurs de pagerank:", pr.sum())


# print("\n"*2)





############### Tests et vérification avec library pagerank ###############

# from fast_pagerank import pagerank_power

# pr=pagerank(A_ex1, 0.85)
# print("\nPagerank:", pr)
# print(pr.sum())

# pr=pagerank_power(A_ex1, p=0.85)
# print("\nVérification pagerank:", pr)


# personalization = np.zeros(A_ex1.shape[0])
# personalization[4] = 1
# pr=personalized_pagerank(A_ex1, personalization, 0.85)
# print("Personnalisation noeud 1:", pr)


# personalization = np.zeros(A_ex1.shape[0])
# personalization[4] = 1
# pr=pagerank_power(A_ex1, p=0.85, personalize=personalization)
# print("Vérification personnalisation noeud 5:", pr)

# print("\n"*2)


