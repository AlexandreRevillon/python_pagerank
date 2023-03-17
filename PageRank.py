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
import argparse
import sys





################### Déclarations des fonctions ###################

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



def import_tsv(filename):
    """Fonction qui importe le fichier tsv et le transforme en liste de listes
    
    Args:
        filename (string): chemin du fichier tsv à importer

    Returns:
        Liste: Liste de listes contenant les données du fichier tsv
    """
    data = list()
    try:
        with open(filename) as fd:
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
    except FileNotFoundError:
        print("Le fichier n'existe pas")
        sys.exit(1)
    return data



def points_et_transitions(filename):
    """Fonction qui renvoie la liste des points et la liste des transitions du graph

     Args:
        filename (string): chemin du fichier tsv à importer
    
    Returns:
        Liste: Liste des points
        Liste: Liste des transitions
    """
    data = import_tsv(filename)
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



def import_adjacence_graph(filename):
    """Fonction qui renvoie la matrice d'adjacence du graphe

     Args:
        filename (string): chemin du fichier tsv à importer
    
    Returns:
        DataFrame: Matrice d'adjacence du graphe
    """
    points, transitions = points_et_transitions(filename)
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
def main(args):
    print("Paramètres choisis:")
    print("\tDumping factor:", args.dumping_factor)
    print("\tTolerance:", args.tolerance)
    print("\tChemin du fichier TSV:", args.tsv_path)
    
    print("\n"*2)
    
    print("Import du graphe...")
    A = import_adjacence_graph(args.tsv_path)
    colnames = A.columns
    print("Graphe importé avec succès")
    
    print("\n"*2)
    
    #Calcul de pagerank non personnalisé
    start = time.time()
    pr = pd.DataFrame(pagerank(A.to_numpy(dtype='float64'), d=args.dumping_factor, tol=args.tolerance), index = colnames)
    pr = pr.sort_values(by=0, ascending=False)
    stop = time.time()

    #Afficgage des résultats
    print("Valeurs de page rank:")
    print(pr.head(args.k_nodes))
    print("\nSomme des valeurs de pagerank: ", pr.sum()[0])
    print("Temps d'exécution: ", stop-start, "secondes")
    
    print("\n"*2)
    
    if args.custom_pagerank:
        #Personalisation des premiers noeud du graphe + calcul du nouveau pagerank
        v = np.zeros(A.shape[0])
        for page in pr.iloc[[0,1,2,3,4]].index:
            v[colnames.get_loc(page)] = 1/5
            
        start = time.time()
        pr = pd.DataFrame(personalized_pagerank(A.to_numpy(dtype='float64'), v), index = colnames)
        pr = pr.sort_values(by=0, ascending=False)
        stop = time.time()
        
        print("Valeurs de page rank avec personnalisation des 5 premières page (selon pagerank non personnalisé):")
        print("Page personnalisées: ", pr.iloc[[0,1,2,3,4]].index)
        print(pr.head(args.k_nodes))
        print("\nSomme des valeurs de pagerank: ", pr.sum()[0])
        print("Temps d'exécution: ", stop-start, "secondes")

        print("\n"*2)

        #Personalisation des dernières noeud du graphe + calcul du nouveau pagerank
        v = np.zeros(A.shape[0])
        last = len(colnames) - 1
        for page in pr.iloc[[last,last-1,last-2,last-3,last-4]].index:
            v[colnames.get_loc(page)] = 1/5
            
        start = time.time()
        pr = pd.DataFrame(personalized_pagerank(A.to_numpy(dtype='float64'), v), index = colnames)
        pr = pr.sort_values(by=0, ascending=False)
        stop = time.time()
        
        print("Valeurs de page rank avec personnalisation des 5 dernières page (selon pagerank non personnalisé):")
        print("Page personnalisées: ", pr.iloc[[last,last-1,last-2,last-3,last-4]].index)
        print(pr.head(args.k_nodes))
        print("\nSomme des valeurs de pagerank: ", pr.sum()[0])
        print("Temps d'exécution: ", stop-start, "secondes")
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gérer les arguments en ligne de commande.")

    parser.add_argument("-d", "--dumping-factor", type=float, default=0.85,
                        help="Le facteur de dumping (entre 0 et 1). Valeur par défaut : 0.85")
    parser.add_argument("-t", "--tolerance", type=float, default=0.0001,
                        help="La tolérance pour les calculs (ex : 0.0001). Valeur par défaut : 0.0001")
    parser.add_argument("-p", "--tsv-path", type=str, required=True,
                        help="Le chemin d'accès du fichier TSV.")
    parser.add_argument("-k", "--k-nodes", type=int, default=10,
                        help="Le nombre de nœuds à afficher.")
    parser.add_argument("-c", "--custom-pagerank", action="store_true",
                        help="Activez cette option pour personnaliser le PageRank. Non activée par défaut.")

    args = parser.parse_args()

    if not (0 < args.dumping_factor < 1):
        print("Erreur : Le facteur de dumping doit être compris entre 0 et 1.")
        sys.exit(1)

    main(args)





############### Tests et vérification avec library pagerank ###############

# from fast_pagerank import pagerank_power

# A = np.array([[0, 1, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0],
#               [1, 1, 0, 0, 1, 0],
#               [0, 0, 0, 0, 1, 1],
#               [0, 0, 0, 1, 0, 1],
#               [0, 0, 0, 1, 0, 0]], dtype='float64')

# pr=pagerank(A, 0.85)
# print("\nPagerank:", pr)
# print(pr.sum())

# pr=pagerank_power(A, p=0.85)
# print("\nVérification pagerank:", pr)


# personalization = np.zeros(A.shape[0])
# personalization[4] = 1
# pr=personalized_pagerank(A, personalization, 0.85)
# print("Personnalisation noeud 5:", pr)


# personalization = np.zeros(A.shape[0])
# personalization[4] = 1
# pr=pagerank_power(A, p=0.85, personalize=personalization)
# print("Vérification personnalisation noeud 5:", pr)

# print("\n"*2)


