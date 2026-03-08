# Prédiction du Taux de Clic : Dataset Avazu

Prédiction de la probabilité qu'un utilisateur clique sur une publicité mobile, sur 1 million d'impressions issues du dataset Avazu. Le projet couvre l'intégralité du pipeline ML, de l'analyse exploratoire jusqu'à une stratégie d'enchères applicable en production.

---

## Problème

Dans la publicité en ligne, à chaque chargement de page une enchère se déroule en quelques millisecondes pour décider quelle publicité afficher. Les annonceurs ont besoin de connaître la probabilité qu'un utilisateur clique sur leur annonce pour décider s'ils enchérissent et combien. C'est le problème de prédiction du taux de clic (CTR).

Les difficultés principales : le dataset est fortement déséquilibré (~17% de clics => class imbalance), les features sont majoritairement des variables catégorielles anonymisées à haute cardinalité, et en production le modèle doit tourner en temps réel.

---

## Dataset

[Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) — 1 million d'impressions publicitaires mobiles sur 11 jours, avec 23 features incluant des informations sur l'appareil, le contexte site/application, et des variables catégorielles anonymisées.

---

## Approche

### Feature Engineering
- Extraction de l'heure de la journée et du jour de la semaine à partir d'un format timestamp propriétaire `YYMMDDHH`
- Construction d'un identifiant utilisateur proxy en combinant device_ip + device_model pour les appareils anonymes
- Fusion de site_id et site_domain en une seule feature site

### Gestion des Variables Catégorielles à Haute Cardinalité
Un encodage one-hot naïf de toutes les features produit plus de 700 000 dimensions, majoritairement du bruit issu de modalités rares. La solution envisagée a été donc :
- **OHE avec filtrage par fréquence** : conservation uniquement des modalités vues plus de 100 fois, soit 1 971 features

### Modèles Comparés

| Modèle | Log Loss | ROC-AUC |
|---|---|---|
| GB + LR Stacking | 0.4001 | 0.7416 |
| XGBoost (256 arbres) | 0.4020 | 0.7390 |
| Régression Logistique (OHE complet) | 0.4015 | 0.7380 |
| XGBoost (50 arbres) | 0.4044 | 0.7342 |
| Gradient Boosting (50 arbres) | 0.4092 | 0.7252 |
| Régression Logistique (features réduites) | 0.4252 | 0.6761 |

### Apprentissage en Ligne
Simulation d'un pipeline de production en streaming avec SGDClassifier.partial_fit(` : le modèle se met à jour par batches de 10 000 impressions sans jamais réentraîner depuis zéro, ce qui correspond au fonctionnement réel des modèles CTR dans les systèmes publicitaires.

### Stratégie d'Enchères
Connexion de la sortie du modèle à un decision maker : étant donné un CTR predicted et une valeur par clic, calcul de la valeur espérée de chaque impression et enchère uniquement lorsqu'elle dépasse le prix minimum. Le modèle sélectionne des impressions significativement plus susceptibles d'être cliquées que le hasard.

---

## Résultats

XGBoost avec 256 estimateurs obtient les meilleures performances. L'analyse de l'importance des features révèle qu'une seule modalité de la feature anonymisée `C16` (probablement la taille de la publicité) est de loin le signal le plus prédictif, devant toute information utilisateur ou site.

Le modèle en apprentissage en ligne atteint des performances comparables à la régression logistique complète sans jamais voir l'intégralité du dataset, ce qui valide l'approche streaming.

---

## Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Scipy, Matplotlib

---

## Exécution

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
jupyter notebook avazu_ctr_project_v2.ipynb
```

Le dataset est téléchargé automatiquement au démarrage du notebook.
