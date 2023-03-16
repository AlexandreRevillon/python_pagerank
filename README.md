# Big data - Projet PageRank



## Strucutre du fichier python

Le fichier python est structuré en 4 partie:
- L'import des librairies
- Les fonctions
- Le programme principal
- Les tests réalisés (cette partie est commantée afin qu'elle ne soit pas exécutée)

## Exécution du programme

Afin de pouvoir exéctuer le programme, il faut en amont avoir installé les librairie utilisée par le fichier python.
Les librairies sont les suivantes : numpy, scipy, igraph et pandas

Pour les installer, il faut exécuter les lignes de code suivantes dans un terminal:

```
pip install numpy
pip install scipy
pip install igraph
pip install pandas
```

Une fois toutes les librairies installée, il suffit de lancer le programme avec la commande suivante:
```
python [path]/PageRank.py
```
Attention à bien remplacer ```\[path\]``` par le chemin d'accès vers le dossier ou se trouve le fichier python.



## Résultats

Ce programme donne en sortie les score de page rank du graph formé par les chemins réalisé par les joueurs de Wikispeedia (uniquement du fichier paths_finished).

Ce programme calcul les valeurs de pagerank pour différentes combinaisons de paramètres.

Il exécute aussi une version personnalisé de l'algorithme de pagerank.
