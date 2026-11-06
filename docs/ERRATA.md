# Errata du rapport et corrections du code

Le [rapport](Rapport_PRE2026_BOURREE_JEAN-BAPTISTE.pdf) est publié **tel qu'il a été remis à
l'ENSTA** (août 2026). En relisant le code ensuite, j'ai trouvé des écarts entre ce que dit le
rapport et ce que faisait réellement le code, ainsi que des erreurs de méthode. Le code du dépôt
est la version corrigée. Cette page liste ces écarts, section par section, puis détaille les
corrections apportées au code.

## 1. Écarts entre le rapport et le code d'origine

| Section du rapport | Ce que dit le rapport | Ce que faisait le code d'origine |
|---|---|---|
| 5.0.1, 7.3 (Fig. 7.4) | α ≈ 1 (défensif) quand le marché chute ; α = 0,10 signifie « marché très haussier ». | **Sens inversé** : α = 0,8 quand ^JKSE est au-dessus de sa moyenne (hausse) et α = 0,1 quand il est en dessous (baisse). L'interprétation de la Fig. 7.4 est donc fausse, et les résultats des sections 5 et 7.4 viennent d'une stratégie « contrarian ». |
| 5.0.1 | Volet défensif : « Markowitz Minimum Variance … maximisant le rendement par unité de risque ». | La phrase se contredit (variance minimale ≠ Sharpe maximal). Le premier hybride utilisait en fait **HRP** comme volet défensif ; la variante `hybride_markowitz` utilisait la variance minimale. |
| 5.0.1 | α réagit à une « forte volatilité ». | L'indicateur compare le **niveau** de l'indice à sa moyenne, pas sa volatilité. |
| 2.1.1, 6.1 | MOSMA travaille dans un espace discret, contrairement à Markowitz, qu'on arrondit. | MOSMA optimisait des **poids continus** puis les arrondissait à l'inférieur ; seuls les frais étaient calculés en lots. |
| 6.2 | MOSMA suit la logique NSGA-II avec distance de congestion ; la répartition irrégulière du front vient de cette distance. | **Aucune distance de congestion** : le classement utilisait la somme `−rendement + risque`. |
| 6.3 | « 100 lignes de référence parfaitement équidistantes ». | Directions **non uniformes** (`t**0.4`). Après suppression des doublons, il restait souvent une dizaine de points. |
| 2.3 | NSGA-III est utile parce que NSGA-II peine « lorsque le nombre d'objectifs dépasse deux ». | Le problème n'a que **2 objectifs**. L'argument ne s'applique pas ici. |
| App (« Hypervolume Plateau ») | Critère d'arrêt sur l'hypervolume. | Le code calculait en réalité l'aire de la **boîte englobante** de l'archive. |
| 4.2 | Le backtest « prélève les frais de transaction virtuels ». | Les backtests de recherche (Markowitz, HRP, Top-K, hybrides) ne prélevaient **aucun frais**. |
| 4.2, 7.4 | Fenêtre glissante de 12 mois. | La fenêtre était de **12 périodes** quelle que soit la fréquence choisie dans l'app : 12 jours en données journalières, 12 semaines en hebdomadaire. |
| 7.4, 9.2 | +60,2 % pour l'hybride, atténué par le biais du survivant. | Ce chiffre vient du **sens inversé d'α**, avec des prix ajustés des dividendes et une seule exécution sans graine aléatoire. Il n'est pas reproductible. |
| 8.2 | « LSTM univariés ». | Le modèle est **multivarié** : prix des 30 actions et indicateurs techniques. |
| 8.2 | MAPE de 2,31 % présenté comme une erreur faible. | Pas de comparaison avec la prédiction naïve, qui fait **mieux** (1,8 %). Le scaler était ajusté sur toutes les données, test compris, et un `bfill` recopiait des prix futurs dans le passé. |
| 4.1 (Fig. 4.1) | « Frontière efficiente discrète ». | La frontière calculée avec cvxpy est **continue**. |
| 4.2 | Le portefeuille 1/N « laisse place au hasard ». | Il n'a rien d'aléatoire : poids égaux, déterministes. |
| Annexe A | Code de `app.py`, `strategies.py` et `backtest_engine.py`. | Ce sont des versions antérieures. Les versions finales s'appelaient `new_app.py`, `new_strat.py` et `new_back.py`, et sont devenues `app.py`, `mosma.py` et `backtest.py` dans ce dépôt. |

## 2. Défauts de forme du rapport

- **Références cassées.** Plusieurs marqueurs `[cite : 10]`, `[cite : 16]` et `[cite : 20]`
  renvoient à des références qui n'existent pas : la bibliographie n'en contient que 4.
- **Fautes de frappe.** « arroundis », « minimim », « otbenir », « intégrand »,
  « L'utilisatueur », « court des actifs » (pour « cours »), « La prévision **est**
  l'allocation » (pour « et »), « poids continues », « LSTM,. ».
- **Figures non reproductibles.** Le code de l'algorithme génétique, de la frontière cvxpy et
  des Fig. 4.1, 5.1 et 6.1 n'a pas été conservé. Les images restent dans [`figures/`](figures/).

## 3. Corrections apportées au code

### Écarts avec le rapport

| # | Problème | Correction |
|---|---|---|
| 1 | α inversé. | Sens conforme au rapport. `regime_switch(..., contrarian=True)` reproduit l'ancien comportement. |
| 2 | MOSMA non discret. | Objectifs évalués sur les lots réellement achetés (génotype continu → phénotype discret). |
| 3 | Pas de classement NSGA-II. | Fronts non dominés + distance de congestion ; archive tronquée par distance de congestion. |
| 4 | Directions NSGA-III non uniformes. | Directions uniformes (`linspace`). |
| 5 | Faux hypervolume. | Hypervolume 2D exact, avec point de référence fixe. |
| 6 | `vc` constant dans [−1, 1], pas de graine. | `vc ∈ [−b, b]` avec `b = 1 − t/T` ; paramètre `seed`. |
| 7 | Backtests de recherche sans frais. | Frais = rotation × frais à chaque rééquilibrage. |

### Backtest et données

| # | Problème | Correction |
|---|---|---|
| 8 | Fenêtre de 12 périodes et rééquilibrage à chaque période. | Fenêtre exprimée en années, rééquilibrage mensuel. |
| 9 | Covariance empirique singulière. | Shrinkage Ledoit-Wolf. |
| 10 | Prix ajustés des dividendes (défaut de yfinance). | `auto_adjust=False`, cohérent avec ^JKSE, qui est un indice de prix. |
| 11 | Toutes les stratégies réutilisaient la position du Hybride. | Chaque stratégie se rééquilibre depuis sa propre position. |
| 12 | Cash négatif possible. | L'allocation garantit un cash ≥ 0. |
| 13 | α calculé sur 5 ans dans l'app et sur 12 périodes dans le backtest. | Même fenêtre partout. |
| 14 | Markowitz de recherche : rendements journaliers + taux sans risque par défaut de la bibliothèque. Sans effet avec PyPortfolioOpt ≥ 1.6 ; avec la 1.5.x, `max_sharpe` échoue et le `except: pass` laisse le portefeuille en cash. | Moments annualisés, taux sans risque explicite, avertissement au lieu d'un échec silencieux. |

### LSTM et application

| # | Problème | Correction |
|---|---|---|
| 15 | LSTM : scaler ajusté sur toutes les données, `bfill`, pas de référence de comparaison, 20 epochs fixes, deux copies du script. | Scaler ajusté sur l'entraînement seul, pas de `bfill`, prédiction naïve comme référence, early stopping, graines fixées, script unique. |
| 16 | App : plantage si aucun portefeuille n'est faisable ; un clic sur l'étoile sélectionnait le portefeuille n°0 ; le backtest était relancé à chaque clic ; les paramètres de l'algorithme n'étaient pas transmis au backtest. | Tous corrigés. Ajout du CAGR, de la volatilité, du Sharpe et des frais payés. |
