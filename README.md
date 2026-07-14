# first-neural-network-from-scratch

## Description

Ce dépôt propose une implémentation de réseaux de neurones entièrement à partir de zéro, en NumPy uniquement, sans recourir à TensorFlow ni PyTorch dans un premier temps. L'objectif est de comprendre en profondeur la mécanique interne d'un réseau, en codant manuellement la propagation avant, la rétropropagation du gradient et la descente de gradient. Le travail est réparti en deux notebooks complémentaires, le premier pose les fondations sur un problème de classification binaire simple, le second étend ces briques à un réseau profond multi-classes sur des données réelles.

Ensuite nous allons améliorer la qualité de notre réseau de neurones en utilisant Pytorch.

## Problème

Une régression logistique ne sait tracer qu'une frontière de décision linéaire, ce qui la limite dès que les classes ne sont pas séparables par un simple plan. Le premier notebook explore si l'ajout d'une couche cachée permet de dépasser cette limite, et à quelles conditions de capacité. Le second notebook répond à une question plus ambitieuse, comment reconnaître automatiquement des chiffres manuscrits parmi dix classes possibles, un problème où la profondeur du réseau, le choix des activations et la formulation multi-classes deviennent indispensables.

## Installation et lancement

Le projet ne nécessite que quelques librairies scientifiques classiques.

```bash
git clone <url-du-repo>
cd first-neural-network-from-scratch
pip install numpy pandas matplotlib plotly scikit-learn notebook
jupyter notebook
```

Pour le premier notebook, le fichier `classification_data.csv` doit rester à la racine du projet. Pour le second, le jeu MNIST est téléchargé automatiquement via scikit-learn au premier lancement. Il suffit d'exécuter les cellules de chaque notebook dans l'ordre pour reproduire l'entraînement et les visualisations associées.

| Notebook | Fichier |
|----------|---------|
| Réseau à une couche cachée | `first_neural_network.ipynb` |
| Réseau profond multi-classes | `deep-neural nework.ipynb` |

## Ce qui a été réalisé

### Notebook 1 — Réseau à deux couches (classification binaire)

Architecture 4 entrées vers 3 neurones cachés vers 1 sortie, soit 15 paramètres au total. Implémentation manuelle de la sigmoïde et de sa dérivée, de l'initialisation aléatoire des poids, de la propagation avant, de l'entropie croisée binaire, de la rétropropagation et de la boucle d'entraînement. Le notebook inclut une comparaison directe avec une régression logistique, des visualisations 3D de la frontière de décision et une étude de l'effet de la taille de la couche cachée.

### Notebook 2 — Réseau profond (classification multi-classes MNIST)

Architecture 784 entrées vers 128 neurones cachés vers 64 neurones cachés vers 10 sorties, soit environ 109 000 paramètres.Le modèle utilise la fonction d'activation ReLU (Rectified Linear Unit) dans les couches cachées afin d'améliorer l'apprentissage des caractéristiques, puis une fonction softmax en sortie pour convertir les scores du réseau en probabilités de classification. L'entraînement repose sur la fonction de perte d'entropie croisée catégorielle, couramment utilisée pour mesurer l'écart entre les prédictions du modèle et les classes attendues.

Le jeu de données MNIST, composé d'images de chiffres manuscrits, est chargé puis prétraité : les valeurs des pixels sont normalisées, les étiquettes sont converties au format one-hot (une représentation binaire des classes) et les données sont séparées en ensembles d'entraînement et de test sur un sous-ensemble de 10 000 images. Les poids du réseau sont initialisés avec la méthode de He, spécialement conçue pour les réseaux utilisant ReLU, avant d'être optimisés par rétropropagation, l'algorithme qui ajuste les poids pour réduire l'erreur de prédiction.

Les performances du modèle sont ensuite évaluées sur le jeu de test à l'aide d'une matrice de confusion, qui visualise les erreurs de classification, ainsi que de la précision par classe. Enfin, une démonstration illustre les prédictions du réseau sur des images individuelles.

## Motivations des choix

Le premier notebook retient la sigmoïde partout, car elle reste la plus simple à dériver quand on découvre la rétropropagation, même si ce n'est pas le choix le plus performant en pratique. Le même jeu de données que la régression logistique est utilisé volontairement pour rendre la comparaison équitable, et la graine aléatoire est fixée à 42 pour garantir la reproductibilité.

Le second notebook adopte les choix qu'un ingénieur machine learning appliquerait sur un vrai problème. ReLU remplace la sigmoïde dans les couches cachées pour limiter la saturation des gradients, l'initialisation He compense la variance introduite par ReLU, softmax convertit les scores en probabilités sur dix classes, et un sous-ensemble de MNIST est retenu pour garder des temps d'entraînement raisonnables en mode batch pur sur CPU. La séparation train/test dès ce stade permet de mesurer la généralisation, ce qui n'était pas encore le cas dans le premier notebook.

## Solution apportée

Les deux notebooks partagent la même philosophie, chaque opération mathématique est explicitée, reliée à sa formule et accompagnée du suivi des dimensions matricielles. Le code est découpé en fonctions distinctes (propagation avant, coût, rétropropagation, entraînement), ce qui permet de modifier un composant isolé, comme la taille d'une couche cachée ou le taux d'apprentissage, pour observer son impact sans toucher au reste de la pipeline.

## Résultats obtenus

Sur le jeu de classification binaire, un réseau à trois neurones cachés plafonne à 64,50 % de précision, nettement en dessous des 97 % de la régression logistique. Ce résultat contre-intuitif s'explique par une couche cachée trop étroite, qui reste bloquée dans un minimum médiocre. L'étude sur la taille de la couche confirme ce diagnostic, les configurations à un, deux ou trois neurones restent à 64,50 %, tandis qu'à partir de cinq neurones le réseau atteint 97,50 %.

Sur MNIST, le réseau profond atteint 86,25 % de précision sur le jeu de test après 50 époques, avec 86,16 % sur l'entraînement, ce qui indique une généralisation correcte sans surapprentissage marqué sur ce sous-ensemble. Les chiffres 0 et 1 sont les mieux reconnus (97,6 % et 94,9 %), tandis que le 5 reste le plus difficile (68,2 %), ce qui se reflète dans la matrice de confusion. Ces résultats restent en deçà des performances obtenues avec des frameworks optimisés et l'ensemble du dataset, mais ils valident que l'implémentation from scratch fonctionne sur un problème réel à dix classes.

## Prochaines étapes

Plusieurs pistes permettraient d'améliorer et de prolonger le projet. 

- [X] Migrer vers PyTorch ou Keras, avec différenciation automatique et accélération GPU
- [ ] Entraîner le réseau profond sur l'intégralité des 70 000 images MNIST et augmenter le nombre d'époques devrait rapprocher les performances des benchmarks usuels.
- [ ] L'ajout d'une régularisation L2 ou de dropout limiterait le surapprentissage sur des architectures plus larges
- [ ] Passage à la descente de gradient par mini-lots améliorerait à la fois la stabilité et la scalabilité
- [ ]  Expérimenter des architectures plus complexes, comme des réseaux convolutionnels mieux adaptés aux images
