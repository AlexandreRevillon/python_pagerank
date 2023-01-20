#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:05:07 2023

@author: arevillo
"""

import numpy as np
import scipy.linalg as nla
import igraph as ig

import csv

data = list()

with open("wikispeedia_paths-and-graph/paths_finished.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        try :
            tmp = list()
            tmp.append(row[0])
            tmp.append(int(row[1]))
            tmp.append(int(row[2]))
            tmp.append(row[3].split(";"))
            tmp.append(row[4])
            data.append(tmp)
        except IndexError :
            tmp = list()

G = ig.Graph()

for ligne in data:
    for i in range(len(ligne[3])-1):
        G.add_vertex(ligne[3][i])
        G.add_vertex(ligne[3][i+1])
        G.add_edge(ligne[3][i], ligne[3][i+1])
    

print(G.Adjacency())









def norm(x):
    somme = 0
    for i in range(x.shape[0]):
        somme += x[i]**2 
    return np.sqrt(somme)


def puissance(A,z0,tol,nitermax):
    q = z0/norm(z0)
    q2 = q
    res = tol + 1
    niter = 0
    z = np.dot(A,q)
    while ((res > tol) and (niter <= nitermax)):
        q = z/norm(z)
        z = np.dot(A,q)
        lam = np.dot(q,z)
        x1 = q
        z2 = np.dot(q2,A)
        q2 = z2/norm(z2)
        y1 = q2
        c = np.dot(y1,x1)
        if c > 5E-2:
            res = norm(z - lam*q)/c
            niter = niter + 1
            nu1=lam
        else:
            print("Problème de convergence !")
            break
    return(nu1,x1)


A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(nla.eigvals(A))

print(puissance(A, np.array([1,1,1]), 0.0000001, 1000))