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
            transitions.add((liste_trans[1], liste_trans[0]))
            while i < len(liste_trans)-1:
                if liste_trans[i+1] == "<":
                    transitions.add((liste_trans[i], liste_trans[i-1]))
                    del liste_trans[i]
                    del liste_trans[i]
                    i -= 1             
                else:
                    transitions.add((liste_trans[i], liste_trans[i+1]))
                    i += 1
            transitions.add((liste_trans[i], liste_trans[i-1]))
    return list(points), list(transitions)


def normalize_dataframe(df):
    """Fonction qui normalise les colonnes d'un dataframe

    Args:
        df (DataFrame): Dataframe dont on veut normaliser les colonnes

    Returns:
        DataFrame: Dataframe dont les colonnes sont normalisées
    """
    df = df.div(df.sum(axis=0), axis=1)
    return df

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

 
       
def pagerank_power_method(P, epsilon=1e-8, max_iter=1000, damp=0.85):
    """Fonction qui calcule le PageRank d'un graphe

    Args:
        P (DataFrame): Matrice de transition normalisée
        epsilon (float, optional): Précision du calcul. Defaults to 1e-8.
        max_iter (int, optional): Nombre maximal d'itérations. Defaults to 1000.
        dump (float, optional): Paramètre de damping. Defaults to 0.85.

    Returns:
        vecteur np.array: PageRank du graphe
    """
    n = P.shape[0]
    v = np.ones(n)/n
    v_old = np.zeros(n)
    i = 0
    while norm(v-v_old) > epsilon and i < max_iter:
        v_old = v
        v = damp*P.dot(v) + (1-damp)/n
        i += 1
    return v
    
    

A = adjacence_graph()

P = normalize_dataframe(A.transpose())

pg = pagerank_power_method(P)

print(pg.sort_values(ascending=False))
print(pg.sum())







# test3 = np.array([[0,1,0],
#                 [0,0,1],
#                 [1,1,0]])

# pg_test3 = pagerank_power_method(normalize_dataframe(pd.DataFrame(test3.transpose())))

# print(pg_test3)
# print(pg_test3.sum())


# test = np.array([[1,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1]])
# test = pd.DataFrame(test)

# print(test)
# print(test.transpose())

# test2 = normalize_dataframe(test.transpose())
# print(test2)
# print(test2.sum(axis=0))