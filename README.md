# first-neural-network-from-scratch

## Description

Ce dépôt propose une implémentation de réseaux de neurones entièrement à partir de zéro, en NumPy uniquement, sans recourir à TensorFlow ni PyTorch dans un premier temps. L'objectif est de comprendre en profondeur la mécanique interne d'un réseau, en codant manuellement la propagation avant, la rétropropagation du gradient et la descente de gradient. Le travail est réparti en notebooks complémentaires, le premier pose les fondations sur un problème de classification binaire simple, le second étend ces briques à un réseau profond multi-classes sur des données réelles.

Ensuite nous améliorons la qualité du modèle avec PyTorch, d'abord via un réseau fully connected, puis via un réseau convolutionnel mieux adapté aux images, un benchmark de régularisation (L2 et Dropout), et enfin une synthèse sur l'intégralité de MNIST.

## Problème

Une régression logistique ne sait tracer qu'une frontière de décision linéaire, ce qui la limite dès que les classes ne sont pas séparables par un simple plan. Le premier notebook explore si l'ajout d'une couche cachée permet de dépasser cette limite, et à quelles conditions de capacité. Les notebooks suivants répondent à une question plus ambitieuse, comment reconnaître automatiquement des chiffres manuscrits parmi dix classes possibles, un problème où la profondeur, le framework, l'architecture convolutionnelle, la régularisation et enfin le volume de données deviennent décisifs.

## Installation et lancement

Le projet ne nécessite que quelques librairies scientifiques classiques.

```bash
git clone <url-du-repo>
cd first-neural-network-from-scratch
pip install numpy pandas matplotlib plotly scikit-learn torch notebook
jupyter notebook
```

Pour le premier notebook, le fichier `classification_data.csv` doit rester à la racine du projet. Pour les notebooks MNIST, le jeu est téléchargé automatiquement via scikit-learn au premier lancement. Il suffit d'exécuter les cellules de chaque notebook dans l'ordre pour reproduire l'entraînement et les visualisations associées.

| Notebook | Fichier |
|----------|---------|
| Réseau à une couche cachée | `first_neural_network.ipynb` |
| Réseau profond multi-classes (NumPy) | `deep-neural_network_numpy.ipynb` |
| Réseau profond multi-classes (PyTorch) | `deep-neural_network_pytorch.ipynb` |
| Réseau convolutionnel (CNN) | `deep-neural_network_cnn.ipynb` |
| Benchmark L2 / Dropout | `deep-neural_network_regularization.ipynb` |
| Synthèse full MNIST | `deep-neural_network_full_mnist.ipynb` |

## Ce qui a été réalisé

### Notebook 1 — Réseau à deux couches (classification binaire)

Architecture 4 entrées vers 3 neurones cachés vers 1 sortie, soit 15 paramètres au total. Implémentation manuelle de la sigmoïde et de sa dérivée, de l'initialisation aléatoire des poids, de la propagation avant, de l'entropie croisée binaire, de la rétropropagation et de la boucle d'entraînement. Le notebook inclut une comparaison directe avec une régression logistique, des visualisations 3D de la frontière de décision et une étude de l'effet de la taille de la couche cachée.

### Notebook 2 — Réseau profond avec NumPy (classification multi-classes MNIST)

Architecture 784 entrées vers 128 neurones cachés vers 64 neurones cachés vers 10 sorties, soit environ 109 000 paramètres. Le modèle utilise la fonction d'activation ReLU dans les couches cachées, puis une fonction softmax en sortie. L'entraînement repose sur l'entropie croisée catégorielle. Le jeu MNIST est chargé puis prétraité (normalisation, one-hot, split train/test sur 10 000 images). Les poids sont initialisés avec He, puis optimisés par rétropropagation. Les performances sont évaluées via matrice de confusion et précision par classe.

### Notebook 3 — Réseau profond avec PyTorch

Même problème et même architecture que le notebook NumPy (784 → 128 → 64 → 10 sur le sous-ensemble MNIST de 10 000 images), mais l'implémentation bascule entièrement sur PyTorch. Le réseau est défini via `nn.Module`, les données passent par un `DataLoader` (mini-lots de taille 64), et la boucle d'entraînement suit le schéma standard : `zero_grad`, forward, `loss.backward()`, `optimizer.step()`. Sur le même jeu de données, le modèle atteint environ 96 % de précision en test, contre ~86 % en batch complet NumPy.

### Notebook 4 — Réseau convolutionnel (CNN)

Même problème MNIST, mêmes données et même protocole d'entraînement que le notebook PyTorch, mais l'architecture devient convolutionnelle : Conv(16) → ReLU → MaxPool → Conv(32) → ReLU → MaxPool → Flatten → FC(128) → FC(10). Le notebook réentraîne aussi le MLP fully connected en baseline pour une comparaison head-to-head, visualise les feature maps, les matrices de confusion et discute les points forts comme les limites du CNN dans ce cas précis.

### Notebook 5 — Benchmark de régularisation (L2 et Dropout)

Même backbone CNN, mêmes données et même protocole, puis comparaison contrôlée de quatre variantes : CNN classique, CNN + L2 (`weight_decay=1e-4`), CNN + Dropout (`p=0.5` sur la couche dense), et CNN + L2 + Dropout. L'objectif est de mesurer si une pénalité de poids plus douce et/ou l'extinction aléatoire de neurones améliorent la généralisation, via la précision test et l'écart train/test.

### Notebook 6 — Synthèse sur MNIST complet

Notebook synthétique qui réentraîne les modèles clés sur l'intégralité de MNIST (60 000 train / 10 000 test, split canonique) : MLP PyTorch, CNN, CNN + L2 et CNN + Dropout. Le protocole est partagé (SGD, `lr=0.1`, 15 époques, batch 128) pour produire un classement final et relire les leçons des notebooks précédents à plus grande échelle.

## Motivations des choix

Le premier notebook retient la sigmoïde partout, car elle reste la plus simple à dériver quand on découvre la rétropropagation. Le second notebook adopte ReLU, He et softmax pour un vrai problème multi-classes. Le troisième conserve volontairement la même architecture pour isoler l'apport du framework. Le quatrième change l'architecture, pas le framework, afin d'isoler l'apport du biais inductif spatial des convolutions. Le cinquième fixe l'architecture CNN et ne fait varier que la régularisation. Le sixième fixe le protocole et fait varier les modèles sur 70 000 images, pour vérifier quelles conclusions tiennent vraiment quand les données ne sont plus un sous-échantillon.

## Solution apportée

Les notebooks NumPy explicitent chaque opération mathématique et suivent les dimensions matricielles. Le notebook PyTorch montre ensuite comment le framework automatise autograd, batching et définition de modèle. Le notebook CNN remplace l'extracteur dense par des filtres partagés. Le notebook de régularisation injecte L2 et Dropout sans changer le backbone. Le notebook de synthèse réutilise ces mêmes briques sur MNIST complet et concentre le diagnostic sur un tableau de bord unique : précision test, écart train/test et temps d'entraînement.

## Résultats obtenus

Sur le jeu de classification binaire, un réseau à trois neurones cachés plafonne à 64,50 % de précision, nettement en dessous des 97 % de la régression logistique, tandis qu'à partir de cinq neurones le réseau atteint 97,50 %.

Sur MNIST, le réseau profond NumPy atteint 86,25 % de précision sur le jeu de test. Avec PyTorch, la même architecture fully connected monte à environ 96 % grâce aux mini-lots et aux biais.

Avec le CNN, sur le sous-ensemble de 10 000 images, la précision de test atteint 98,10 %, contre 95,85 % pour le MLP PyTorch, soit un gain d'environ +2,25 points.

Sur le benchmark de régularisation (30 époques, sous-ensemble 10k), le CNN sans régularisation reste le meilleur en test final (98,25 %). L2 et Dropout n'améliorent pas clairement le score final sur ce sous-échantillon déjà très propre.

Sur MNIST complet (60k/10k, 15 époques), le classement change :

| Modèle | Train | Test | Gap |
|--------|-------|------|-----|
| MLP | 98,92 % | 97,51 % | +1,41 % |
| CNN | 99,23 % | 98,55 % | +0,68 % |
| CNN + L2 | 99,13 % | 98,58 % | +0,55 % |
| CNN + Dropout | 99,50 % | 99,03 % | +0,47 % |

Le gagnant est CNN + Dropout (99,03 % en test, plus petit gap). Le CNN bat clairement le MLP (+1,04 à +1,52 points), et avec davantage de données le Dropout devient utile, alors qu'il ne l'était pas clairement sur le sous-ensemble de 10k. La leçon principale est donc que l'intérêt d'une régularisation dépend du régime de données, pas seulement de l'architecture.

## Points forts et limites du CNN (cas MNIST)

Points forts : les convolutions respectent la géométrie locale des chiffres, le partage de filtres apprend des traits réutilisables, et le modèle extrait une hiérarchie de features (bords → parties de chiffres → classe) plus adaptée aux images qu'un vecteur plat de 784 pixels.

Limites : MNIST est déjà un benchmark facile, le gain absolu reste donc borné, une CNN standard n'est pas invariante aux fortes rotations ou changements d'échelle sans augmentation, elle reste plus coûteuse à exécuter sur CPU, et ce n'est pas le bon outil pour des données tabulaires sans structure spatiale.

## Prochaines étapes

- [X] Migrer vers PyTorch, avec différenciation automatique et accélération GPU
- [X] Passage à la descente de gradient par mini-lots
- [X] Expérimenter une architecture convolutionnelle mieux adaptée aux images
- [X] Ajouter une régularisation L2 ou du dropout
- [X] Entraîner sur l'intégralité des 70 000 images MNIST
- [ ] Ajouter de l'augmentation de données (translations, petites rotations)
- [ ] Tester BatchNorm et des CNN plus profonds, ou du transfer learning sur des jeux plus difficiles
- [ ] Balayer plus finement `weight_decay` et `dropout_p` pour trouver un régime utile
