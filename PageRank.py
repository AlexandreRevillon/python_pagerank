#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:05:07 2023

@author: arevillo
"""

import numpy as np
import scipy.linalg as nla
import igraph as ig
import pandas as pd
import csv

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
    
    
    
        
def l1(x):
    """Fonction qui calcule la norme L1 d'un vecteur

    Args:
        x (array): Vecteur dont on veut calculer la norme L1

    Returns:
        array: Norme L1 du vecteur
    """
    return np.sum(np.abs(x))
    
def get_google_matrix(A, d=0.85):
    """Fonction qui renvoie la matrice de Google

    Args:
        A (array): Matrice d'adjacence du graphe
        d (float, optional): damping factor. Defaults to 0.15.

    Returns:
        array: Matrice de Google
    """
    n = A.shape[0]
    A = A.transpose()
    
    # for sink nodes
    is_sink = np.sum(A, axis=0) == 0
    B = (np.ones_like(A) - np.identity(n)) / (n-1)
    A[:, is_sink] += B[:, is_sink]
    
    D_inv = np.diag(1/np.sum(A, axis=0))
    M = np.dot(A, D_inv) 
    
    # for disconnected components
    M = (1-d)*M + d*np.ones((n,n))/n
    
    return M    
    
    
def pagerank_power(A, d=0.85, max_iter=100, eps=1e-9):
    """Fonction qui calcule le vecteur de PageRank

    Args:
        A (array): Matrice d'adjacence du graphe
        d (float, optional): damping factor. Defaults to 0.85.
        max_iter (int, optional):nombre d'itérations maximal. Defaults to 100.
        eps (float, optional): tolerance. Defaults to 1e-9.

    Returns:
        array: vecteur de PageRank
    """
    M = get_google_matrix(A, d=d)
    n = A.shape[0]
    V = np.ones(n)/n
    for _ in range(max_iter):
        V_last = V
        V = np.dot(M, V)
        if  l1(V-V_last)/n < eps:
            return V
    return V




A = adjacence_graph()
colnames = A.columns

pg = pd.DataFrame(pagerank_power(A.to_numpy(dtype='float64')), index = colnames)

print(pg.sort_values(by=0, ascending=False))
print(pg.sum())



# A_ex1 = np.array([[0, 1, 1, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0],
#                   [1, 1, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 1, 1],
#                   [0, 0, 0, 1, 0, 1],
#                   [0, 0, 0, 1, 0, 0]], dtype='float64')

# pg_ex1 = pagerank_power(A_ex1)
# print(pg_ex1)


# ex= np.array([[1., 0., 0., 0., 0., 0., 0., 1.],
#             [0., 1., 0., 0., 1., 0., 0., 0.],
#             [1., 1., 0., 0., 0., 0., 0., 0.],
#             [0., 0., 1., 0., 0., 0., 0., 1.],
#             [0., 1., 1., 0., 0., 0., 0., 0.],
#             [0., 1., 0., 0., 1., 0., 0., 0.],
#             [1., 1., 0., 0., 0., 0., 0., 0.],
#             [0., 1., 1., 0., 0., 0., 0., 0.]])

# print(pagerank_power(ex))


# test3 = np.array([[0, 1, 0, 0], 
#                   [1, 0, 1, 0], 
#                   [0, 0, 0, 1], 
#                   [0, 1, 0, 0]])

# pg_test3 = pagerank_power_method(test3)
# print(pg_test3)
# print(pg_test3.sum())


# test = np.array([[1,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1]])
# test = pd.DataFrame(test)

# print(test)
# print(test.transpose())

# test2 = normalize_dataframe(test.transpose())
# print(test2)
# print(test2.sum(axis=0))