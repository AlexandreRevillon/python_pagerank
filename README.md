# README.md

## Introduction
Ce programme est conçu pour calculer le PageRank des pages parcourus par les joueurs de Wikispeedia à partir d'un fichier TSV des chemins réalisé. Il permet également de personnaliser le PageRank à l'aide de vecteurs de personnalisation. Le programme prend en charge plusieurs arguments pour personnaliser la tolérance, le facteur de dumping, le nombre de nœuds à afficher et le chemin d'accès au fichier TSV.

## Prérequis
Assurez-vous d'avoir installé les bibliothèques suivantes pour pouvoir exécuter le programme :
- numpy
- scipy
- igraph
- pandas
Vous pouvez les installer à l'aide de pip:

```
pip install numpy scipy python-igraph pandas
```

## Utilisation
Pour utiliser le programme, exécutez le fichier Python avec les arguments appropriés.

```
python <chemin_nom_du_fichier>.py --tsv-path <chemin_du_fichier_tsv> [options]
```
ou
```
python <chemin_nom_du_fichier>.py -p <chemin_du_fichier_tsv> [options]
```

Les options disponibles sont les suivantes :

-d ou --dumping-factor : Le facteur de dumping (entre 0 et 1). Valeur par défaut : 0.85
-t ou --tolerance : La tolérance pour les calculs (ex : 0.0001). Valeur par défaut : 0.0001
-p ou --tsv-path : Le chemin d'accès du fichier TSV. (Obligatoire)
-k ou --k-nodes : Le nombre de nœuds à afficher. Valeur par défaut : 10
-c ou --custom-pagerank : Activez cette option pour personnaliser le PageRank. Non activée par défaut.

### Exemple
```
python <nom_du_fichier>.py --tsv-path data.tsv -d 0.9 -t 0.00001 -k 20 -c
```
Cet exemple utilise un facteur de dumping de 0.9, une tolérance de 0.00001, affiche 20 nœuds et active la personnalisation du PageRank.

## Fonctions principales
Le programme contient les fonctions suivantes :

norm(x): Calcule la norme euclidienne d'un vecteur.
import_tsv(filename): Importe le fichier TSV et le transforme en liste de listes.
points_et_transitions(filename): Renvoie la liste des points et la liste des transitions du graphe.
import_adjacence_graph(filename): Renvoie la matrice d'adjacence du graphe.
pagerank(matrice_adjacence, d=0.85, max_iter=100, tol=1e-6): Calcule les valeurs PageRank d'une matrice d'adjacence.
personalized_pagerank(matrice_adjacence, v, d=0.85, max_iter=100, tol=1e-6): Calcule les valeurs PageRank d'une matrice d'adjacence à partir d'un vecteur de personnalisation.
Licence
Ce programme est distribué sous la licence MIT.


## Résultats

Ce programme donne en sortie les scores de page rank des ```k noeuds``` du graph.

Lorsque l'option de personnalisation est activé, le programme donne ensortie des même scores, ainsi que les  ```k noeuds``` de 2 personnalisations distinctes:
- personnalisation des 5 premières pages (selon le page rank non personnalisée)
- personnalisation des 5 dernières pages (selon le page rank non personnalisée)

