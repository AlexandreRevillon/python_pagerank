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

data = list()

with open("wikispeedia_paths-and-graph/paths_finished.tsv") as fd:
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

G = ig.Graph(directed=True)
test = G.get_vertex_dataframe()

points = set()
transitions = set()
for ligne in data:
    for i in range(len(ligne[3])):
        points.add(ligne[3][i])
    for i in range(len(ligne[3])-1):
        if(ligne[3][i] != "<"):
            j=0
            k=0
            while ligne[3][i+j+1] == "<":
                j+=1
                while ligne[3][i-j-k] == "<":
                    k+=1
            if ligne[3][i-j-2*k] == ligne[3][i+j+1]:
                print("erreur ({},{}), ligne n° {}".format(ligne[3][i-j-2*k],ligne[3][i+j+1], ligne[0]))
            transitions.add((ligne[3][i-j-2*k],ligne[3][i+j+1]))

points = list(points)
for p in points:
    G.add_vertex(name=p)

G.add_edges(list(transitions))

A = pd.DataFrame(G.get_adjacency().data, columns=points, index=points)


# for i in range(len(T)):
#     for j in range(len(T[i])):
#         if T[i][j] >1 or T[i][j] < 0:
#             print("erreur ({},{})".format(i, j))


def norm(x):
    somme = 0
    for i in range(x.shape[0]):
        somme += x[i]**2
    return np.sqrt(somme)


def puissance(A, z0, tol, nitermax):
    q = z0/norm(z0)
    q2 = q
    res = tol + 1
    niter = 0
    z = np.dot(A, q)
    while ((res > tol) and (niter <= nitermax)):
        q = z/norm(z)
        z = np.dot(A, q)
        lam = np.dot(q, z)
        x1 = q
        z2 = np.dot(q2, A)
        q2 = z2/norm(z2)
        y1 = q2
        c = np.dot(y1, x1)
        if c > 5E-2:
            res = norm(z - lam*q)/c
            niter = niter + 1
            nu1 = lam
        else:
            print("Problème de convergence !")
            break
    return(nu1, x1)


B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(nla.eigvals(B))

print(puissance(B, np.array([1, 1, 1]), 0.0000001, 1000))







