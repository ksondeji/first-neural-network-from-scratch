# first-neural-network-from-scratch

## Description

Ce projet implémente un réseau de neurones à deux couches entièrement à partir de zéro, en NumPy uniquement, sans recourir à une librairie de deep learning comme TensorFlow ou PyTorch. L'objectif est de comprendre en profondeur la mécanique interne d'un réseau, en codant manuellement la propagation avant, la rétropropagation du gradient et la descente de gradient. Le tout est présenté dans un notebook pédagogique qui détaille chaque étape, suit les dimensions des matrices à chaque calcul et visualise les résultats.

## Problème

Une régression logistique ne sait tracer qu'une frontière de décision linéaire, ce qui la limite dès que les classes ne sont pas séparables par un simple plan. Ce projet répond à cette limite en ajoutant une couche cachée, qui permet au modèle d'apprendre une frontière de décision non linéaire. La question concrète posée tout au long du notebook est donc la suivante:

Une couche cachée apporte-t-elle réellement un gain sur un problème de classification binaire, si oui, à quelles conditions?

## Installation et lancement

Le projet ne nécessite que quelques librairies scientifiques classiques.

```bash
git clone <url-du-repo>
cd first-neural-network-from-scratch
pip install numpy pandas matplotlib plotly notebook
jupyter notebook first_neural_network.ipynb
```

Le fichier `classification_data.csv` doit rester à la racine, car le notebook le charge directement. Il suffit ensuite d'exécuter les cellules dans l'ordre pour reproduire l'entraînement et l'ensemble des visualisations.

## Ce qui a été réalisé

Le réseau suit une architecture 4 entrées vers 3 neurones cachés vers 1 sortie, biais inclus, soit 15 paramètres au total. Les briques suivantes ont été codées à la main, la fonction d'activation sigmoïde et sa dérivée, l'initialisation aléatoire des poids, la propagation avant à travers les deux couches, la fonction de coût d'entropie croisée binaire, la rétropropagation via la règle de la chaîne et la boucle d'entraînement par descente de gradient. Le notebook ajoute aussi une comparaison directe avec une régression logistique, une visualisation 3D des prédictions et de la frontière de décision, ainsi qu'une étude de l'effet de la taille de la couche cachée.

## Motivations des choix

La sigmoïde a été retenue pour les deux couches, car elle reste l'activation la plus simple à dériver et la plus parlante quand on découvre la rétropropagation, même si ce n'est pas le choix le plus performant en pratique. L'initialisation des poids se fait avec de petites valeurs aléatoires, pour éviter que la sigmoïde ne sature dès le départ et n'annule les gradients. La graine aléatoire est fixée à 42, afin de garantir des résultats reproductibles. Enfin, le même jeu de données que la régression logistique (issue d'une précédente analyse) est utilisé volontairement, ce qui rend la comparaison entre les deux modèles parfaitement équitable.

## Solution apportée

La solution est un réseau de neurones complet et lisible, où chaque opération mathématique est explicitée et reliée à sa formule. La descente de gradient utilise un taux d'apprentissage de 1.0 sur 2000 itérations, en mode batch sur les 200 exemples. La structure du code, fonctions séparées pour la propagation avant, le coût, la rétropropagation et l'entraînement, permet de modifier facilement un élément isolé, comme la taille de la couche cachée, pour observer son impact.

## Résultats obtenus

Avec trois neurones cachés, le réseau plafonne à 64,50 % de précision, soit nettement en dessous des 97 % de la régression logistique sur le même jeu de données. Ce résultat contre-intuitif s'explique par une couche cachée trop étroite, qui ne dispose pas d'assez de capacité pour exploiter sa non-linéarité et reste bloquée dans un minimum médiocre. L'étude sur la taille de la couche cachée confirme ce diagnostic, les configurations à un, deux ou trois neurones restent à 64,50 %, tandis que cinq neurones atteignent 97,50 % et dix neurones se maintiennent à ce niveau. La leçon principale est donc que la profondeur seule ne suffit pas, la largeur de la couche cachée est un facteur déterminant pour qu'un réseau dépasse un modèle linéaire.

## Prochaines étapes

Plusieurs pistes permettraient d'améliorer le projet :

- [ ] remplacer la sigmoïde des couches cachées par ReLU ou tanh pour accélérer l'apprentissage et atténuer la saturation des gradients
- [ ] ajouter des couches supplémentaires pour passer à un véritable réseau profond, et introduire la classification multi-classes avec une sortie softmax
- [ ] Il serait aussi pertinent de séparer les données en jeu d'entraînement et jeu de test pour mesurer la généralisation
- [ ] ajouter une régularisation pour limiter le surapprentissage
- [ ] remplacer la descente de gradient batch par une version par mini-lots afin de mieux passer à l'échelle sur des jeux de données plus volumineux.
