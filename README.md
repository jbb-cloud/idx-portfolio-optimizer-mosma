# IDX Portfolio Optimizer : MOSMA en lots entiers

Optimisation de portefeuille multi-objectifs pour la **Bourse d'Indonésie (IDX)**, sous les
contraintes réelles du marché : achat par **lots de 100 actions**, **frais asymétriques**, et
rééquilibrage à partir du portefeuille déjà détenu.

Projet réalisé pendant mon stage de PRE (ENSTA Paris, été 2026) au département P2MS de
l'Institut Teknologi Bandung (ITB), sous la direction de Bouh Adhe.

📄 [Rapport de stage](docs/Rapport_PRE2026_BOURREE_JEAN-BAPTISTE.pdf) ·
⚠️ [Faiblesses des modèles](docs/LIMITES.md) ·
📝 [Errata du rapport et corrections](docs/ERRATA.md)

![Front de Pareto dans l'application](docs/figures/app_pareto_v2.png)

## Ce que fait l'outil

L'utilisateur indique son budget, ses frais de courtage, un poids maximum par action et ses
positions actuelles. L'application renvoie alors :

- un **front de Pareto** rendement/risque où chaque point est un portefeuille **réellement
  achetable** en lots entiers, avec son plan d'ordres (« acheter 22 lots d'ADRO.JK… »), les
  frais et le cash restant ;
- un portefeuille **hybride** qui mélange le portefeuille le plus agressif et le plus défensif
  du front selon le régime de marché, mesuré sur l'indice ^JKSE ;
- un **backtest walk-forward** de ces stratégies, comparé au portefeuille équipondéré (1/N) et
  à l'indice.

## Méthode en bref

| Étape | Choix |
|---|---|
| Estimation | Moyenne historique annualisée ; covariance Ledoit-Wolf ; fenêtre glissante de 1 à 3 ans |
| Optimisation | MOSMA (*Multi-Objective Slime Mould Algorithm*). Chaque individu est un vecteur de poids converti en lots entiers avant d'être évalué. Objectifs : −rendement net de frais et volatilité |
| Classement | NSGA-II : fronts non dominés, puis distance de congestion ; archive externe des solutions non dominées |
| Front affiché | Sélection NSGA-III avec directions de référence uniformes ; arrêt possible sur plateau de l'hypervolume |
| Hybride | `w = (1 − α)·w_MaxReturn + α·w_MinRisk`, avec α = 0,1 / 0,5 / 0,8 selon la position de ^JKSE par rapport à sa moyenne ± 0,5σ |
| Backtest | Départ en cash, rééquilibrage mensuel, uniquement des données passées ; lots, cash et frais suivis séparément pour chaque stratégie |

Le détail mathématique est dans le rapport (chapitres 2 à 7).

## Résultats et limites

Les résultats sont **décevants, et c'est une conclusion importante du projet**.

![Backtest corrigé](docs/figures/result_backtest_corrected.png)

- **Backtest.** Sur 10/2023 → 11/2026, avec frais, la stratégie hybride corrigée fait −18,9 %.
  Elle limite la perte par rapport à ^JKSE (−25,2 %) mais ne bat pas le simple 1/N (−17,3 %).
  Le +60 % annoncé dans le rapport venait d'un **α inversé** dans le code d'origine (−44,4 %
  sur cette période). Le classement des stratégies change dès qu'on retire une seule action de
  l'univers.
- **LSTM.** Le modèle prédit **moins bien que la prédiction naïve** « demain = aujourd'hui »
  (MAPE de 2,4 à 3 %, contre 1,8 % pour le naïf). Il n'a pas été intégré à l'application.
- **Biais.** Le backtest porte sur l'IDX30 actuel, donc uniquement sur des entreprises qui ont
  survécu, avec un seul historique, sans dividendes et sans test statistique.

![LSTM contre prédiction naïve](docs/figures/result_lstm_vs_naive.png)

La page **[Faiblesses des modèles](docs/LIMITES.md)** détaille, pour Markowitz, l'algorithme
génétique, MOSMA, NSGA, l'hybride, les frais, le backtest et le LSTM, ce que chaque modèle
suppose, où il casse et ce qui a été observé.

Le code a été **corrigé après la remise du rapport**. Le rapport est publié tel quel, et
[ERRATA.md](docs/ERRATA.md) liste les écarts section par section, ainsi que les 16 corrections
apportées.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Scripts de recherche : `pip install -r requirements-research.txt`, puis :

```bash
python research/make_figures.py          # backtest corrigé -> docs/figures/
python research/lstm_prediction.py --save # LSTM contre prédiction naïve
```

Les données viennent de Yahoo Finance (`yfinance`) et sont téléchargées à l'exécution : les
résultats dépendent donc de la date. Le paramètre `seed` fixe le hasard de MOSMA.

## Structure

```
app.py                     Application Streamlit
mosma.py                   MOSMA, classement NSGA-II, archive, sélection NSGA-III, hypervolume
backtest.py                Candidats, indicateur de régime, backtest walk-forward, métriques
utils.py                   Données, covariance Ledoit-Wolf, frais, conversion poids -> lots
research/
  classic_backtests.py     Backtests du rapport (chap. 4-5) : Markowitz, HRP, Top-K, 1/N, hybrides
  lstm_prediction.py       Expérience LSTM (chap. 8), comparée à la prédiction naïve
  make_figures.py          Figures de résultats du README
docs/
  Rapport_PRE2026_...pdf   Rapport de stage (version remise)
  LIMITES.md               Faiblesses de chaque modèle
  ERRATA.md                Écarts entre rapport et code, corrections
  figures/                 Figures du rapport, captures de l'app, résultats corrigés
```

## Droits

Ce dépôt est public à titre de **démonstration**. Aucune licence n'est accordée : tous droits
réservés. Pour toute réutilisation du code ou du rapport, contactez-moi.

*Prototype de recherche : ce n'est pas un conseil en investissement.*
