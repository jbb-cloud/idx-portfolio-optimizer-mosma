"""Research backtests used for chapters 4 and 5 of the report
(Markowitz, HRP, Top-K momentum, naive 1/N and the hybrid strategies).

Monthly walk-forward: weights are estimated on a trailing window ending at
`current_end` and held for the next month (buy-and-hold inside the month).

Fixes compared with the internship version:
* transaction costs are now deducted at every rebalancing (turnover x fee),
  the report (section 4.2) said they were but the code did not;
* Markowitz now receives *annualised* mean/covariance and an explicit
  annual risk-free rate. Before, daily means were passed with the library's
  default risk-free rate: fine with PyPortfolioOpt >= 1.6 (default 0), but
  with 1.5.x (default 0.02, annual) max_sharpe() fails and the silent
  `except: pass` left the strategy in cash;
* solver errors are no longer silently swallowed (a warning is printed);
* regime_switch follows the report (alpha = weight of the defensive leg,
  high in a downtrend). `CONTRARIAN = True` reproduces the original code.
"""
import warnings
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy
import pypfopt as pf
import pandas as pd
import numpy as np
from pypfopt.hierarchical_portfolio import HRPOpt

ANN_FACTOR = 252          # daily data
RISK_FREE_RATE = 0.05     # annual
FRAIS_ACHAT = 0.0015
FRAIS_VENTE = 0.0025
CONTRARIAN = False


def frais_rotation(w_new, w_old):
    """Cost of moving from w_old to w_new, as a fraction of wealth."""
    d = np.asarray(w_new, dtype=float) - np.asarray(w_old, dtype=float)
    return FRAIS_ACHAT * np.clip(d, 0, None).sum() + FRAIS_VENTE * np.clip(-d, 0, None).sum()


def changer_date(start_str, nbr_month):
    start_date = pd.to_datetime(start_str)
    new_date = start_date + pd.DateOffset(months=nbr_month)
    return new_date.strftime("%Y-%m-%d")

def rendement_reel_oos(weights, data_test):
    if data_test.empty or len(data_test) < 2:
        return 0 
    rendements_actifs = (data_test.iloc[-1] / data_test.iloc[0]) - 1
    rendement_port = weights.T @ rendements_actifs.to_numpy()
    return rendement_port

def regime_switch(start, end, data_indice):
    """Weight of the defensive leg: 0.1 in an uptrend, 0.8 in a downtrend."""
    indice_train = data_indice.loc[start:end]
    if isinstance(indice_train, pd.DataFrame):
        indice_train = indice_train.squeeze(axis=1)
    if len(indice_train) < 2:
        return 0.5
    barriere = float(indice_train.std()) * 0.5
    current_price = float(indice_train.iloc[-1])
    average = float(indice_train.mean())
    if current_price > average + barriere:
        alpha = 0.1
    elif current_price < average - barriere:
        alpha = 0.8
    else:
        return 0.5
    return 0.9 - alpha if CONTRARIAN else alpha

def hrp_weight(data, MOIS_TEST, MOIS_HRP, current_end, assets):
    rendements_reels_mensuels_OPTI = []
    poids_prec = np.zeros(len(assets))
    current_start = changer_date(current_end, -MOIS_HRP)
    
    for i in range(MOIS_TEST):
        # A. Données d'entraînement
        data_train = data.loc[current_start:current_end]
        poids_optimaux = np.zeros(len(assets)) # Par défaut: 100% Cash (0% partout)
        
        if not data_train.empty and len(data_train) > 1:
            # filtre momentum
            rendement_periode = (data_train.iloc[-1] / data_train.iloc[0]) - 1
            action_gagnante = rendement_periode[rendement_periode > 0].index.tolist()
            data_train = data_train[action_gagnante]
    
            # On ne lance HRP que s'il reste des actions gagnantes
            if not data_train.empty and len(data_train.columns) > 1:
                returns_train = data_train.pct_change().dropna()
                try:
                    hrp = HRPOpt(returns_train)
                    raw_weights = hrp.optimize()
                    poids_optimaux = np.array([raw_weights.get(ticker, 0) for ticker in assets])
                except Exception as e:
                    warnings.warn(f"HRP failed at {current_end}: {e}")
    
        # D. Données de test
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
    
        # E. Calcul OOS 
        if not data_test.empty:
            ret_oos_OPTI = rendement_reel_oos(poids_optimaux, data_test)
        else:
            ret_oos_OPTI = 0 
        
        ret_oos_OPTI -= frais_rotation(poids_optimaux, poids_prec)
        poids_prec = poids_optimaux
        rendements_reels_mensuels_OPTI.append(ret_oos_OPTI)
    
        # F. Glissement
        current_start = changer_date(current_start, 1)
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_OPTI
    
def top_K(data, MOIS_TEST, MOIS_TOP_Kassets, current_end, assets, K):
    rendements_reels_mensuels_OPTI = []
    poids_prec = np.zeros(len(assets))
    current_start = changer_date(current_end, -MOIS_TOP_Kassets)
    
    for i in range(MOIS_TEST):
        data_train = data.loc[current_start:current_end]
        poids_opt = np.zeros(len(assets))
        
        if not data_train.empty and len(data_train) > 1:
            rendement = (data_train.iloc[-1] / data_train.iloc[0]) - 1
            top_actions = rendement.sort_values(ascending=False).head(K)
            top_actions = top_actions[top_actions > 0]
            
            if len(top_actions) > 0:
                weight = 1 / len(top_actions)
                for ticker in top_actions.index:
                    if ticker in assets:
                        idx = assets.index(ticker)
                        poids_opt[idx] = weight

        # D. Données de test
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
    
        # E. Calcul OOS 
        if not data_test.empty:
            ret_oos_OPTI = rendement_reel_oos(poids_opt, data_test)
        else:
            ret_oos_OPTI = 0
        
        ret_oos_OPTI -= frais_rotation(poids_opt, poids_prec)
        poids_prec = poids_opt
        rendements_reels_mensuels_OPTI.append(ret_oos_OPTI)
        
        # F. Glissement
        current_start = changer_date(current_start, 1)
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_OPTI

def naif(data, MOIS_TEST, current_end, assets):
    rendements_reels_mensuels_NAIF = []
    poids_prec = np.zeros(len(assets))
    poids_naifs = np.full(len(assets), 1.0 / len(assets))
    
    for i in range(MOIS_TEST):
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
        
        if not data_test.empty:
            ret_oos_NAIF = rendement_reel_oos(poids_naifs, data_test)
        else:
            ret_oos_NAIF = 0
            
        ret_oos_NAIF -= frais_rotation(poids_naifs, poids_prec)  # only the first month pays
        poids_prec = poids_naifs
        rendements_reels_mensuels_NAIF.append(ret_oos_NAIF)
        
        # Ici, pas besoin de current_start, on décale juste current_end
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_NAIF

def hybride(data, MOIS_TEST, MOIS_TOP_K, MOIS_HRP, current_end, assets, K, data_indice):
    rendements_reels_mensuels_OPTI = []
    poids_prec = np.zeros(len(assets))
    
    for i in range(MOIS_TEST):
        # 🟢 CRÉATION DE DEUX FENÊTRES DE TEMPS DISTINCTES
        start_top_k = changer_date(current_end, -MOIS_TOP_K)
        start_hrp = changer_date(current_end, -MOIS_HRP)
        
        data_train_top_k = data.loc[start_top_k:current_end]
        data_train_hrp = data.loc[start_hrp:current_end]
        
        poids_opt_hrp = np.zeros(len(assets))
        poids_opt_top_K = np.zeros(len(assets))
        
        # --- 1. Stratégie TOP K (Avec sa fenêtre de 6 mois) ---
        if not data_train_top_k.empty and len(data_train_top_k) > 1:
            rendement_top_k = (data_train_top_k.iloc[-1] / data_train_top_k.iloc[0]) - 1
            top_actions = rendement_top_k.sort_values(ascending=False).head(K)
            top_actions = top_actions[top_actions > 0]
            
            if len(top_actions) > 0:
                weight = 1 / len(top_actions)
                for ticker in top_actions.index:
                    if ticker in assets:
                        idx = assets.index(ticker)
                        poids_opt_top_K[idx] = weight
                        
        # --- 2. Stratégie HRP (Avec sa fenêtre de 12 mois) ---
        if not data_train_hrp.empty and len(data_train_hrp) > 1:
            rendement_hrp = (data_train_hrp.iloc[-1] / data_train_hrp.iloc[0]) - 1
            actions_gagnantes = rendement_hrp[rendement_hrp > 0].index.tolist()
            data_train_filtre = data_train_hrp[actions_gagnantes]
            
            if not data_train_filtre.empty and len(data_train_filtre.columns) > 1:
                returns_train = data_train_filtre.pct_change().dropna()
                try:
                    hrp = HRPOpt(returns_train)
                    raw_weights = hrp.optimize()
                    poids_opt_hrp = np.array([raw_weights.get(ticker, 0) for ticker in assets])
                except Exception as e:
                    warnings.warn(f"Optimiser failed at {current_end}: {e}")

        # --- D. Données de test ---
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
    
        # --- E. Calcul OOS ---
        if not data_test.empty:
            ret_oos_OPTI_top_K = rendement_reel_oos(poids_opt_top_K, data_test)
            ret_oos_OPTI_hrp = rendement_reel_oos(poids_opt_hrp, data_test)
        else:
            ret_oos_OPTI_top_K = 0
            ret_oos_OPTI_hrp = 0
            
        # On utilise la fenêtre longue (HRP) pour définir le régime de l'indice de marché
        alpha = regime_switch(start_hrp, current_end, data_indice)
        
        # Moyenne pondérée
        rendement_hybride_du_mois = (ret_oos_OPTI_top_K * (1-alpha) )+ (ret_oos_OPTI_hrp * alpha)
        poids_hyb = (1 - alpha) * poids_opt_top_K + alpha * poids_opt_hrp
        rendement_hybride_du_mois -= frais_rotation(poids_hyb, poids_prec)
        poids_prec = poids_hyb
        rendements_reels_mensuels_OPTI.append(rendement_hybride_du_mois)
        
        # --- F. Glissement ---
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_OPTI

def markowitz(data, MOIS_TEST, MOIS_MARKOWITZ, current_end, assets):
    rendements_reels_mensuels_OPTI = []
    poids_prec = np.zeros(len(assets))
    current_start = changer_date(current_end, -MOIS_MARKOWITZ) # Fenêtre arbitraire de 12 mois pour l'entraînement
    
    for i in range(MOIS_TEST):
        data_train = data.loc[current_start:current_end]
        poids_optimaux = np.zeros(len(assets))
        
        if not data_train.empty and len(data_train) > 1:
            returns_train = data_train.pct_change().dropna()
            
            if not returns_train.empty:
                try:
                    mu = returns_train.mean() * ANN_FACTOR
                    sigma = returns_train.cov() * ANN_FACTOR
                    ef = pf.EfficientFrontier(mu, sigma)
                    raw_weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
                    poids_optimaux = np.array([raw_weights.get(ticker, 0) for ticker in assets])
                except Exception as e:
                    warnings.warn(f"Optimiser failed at {current_end}: {e}")
        
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
        
        if not data_test.empty:
            ret_oos_OPTI = rendement_reel_oos(poids_optimaux, data_test)
        else:
            ret_oos_OPTI = 0
            
        ret_oos_OPTI -= frais_rotation(poids_optimaux, poids_prec)
        poids_prec = poids_optimaux
        rendements_reels_mensuels_OPTI.append(ret_oos_OPTI)
        
        current_start = changer_date(current_start, 1)
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_OPTI


def hybride_markowitz(data, MOIS_TEST, MOIS_TOP_K, MOIS_MARKOWITZ, current_end, assets, K, data_indice, weight_limit=0.20):
    """
    Variante de hybride() où :
    - le volet "agressif" (ex Top-K momentum) est remplacé par un Markowitz
      max Sharpe, contraint à weight_limit (20% par défaut) par actif, et
      restreint aux K (par défaut 5) meilleures actions en momentum sur la
      fenêtre MOIS_TOP_K (au lieu d'une simple équipondération du Top-K).
    - le volet "défensif" (ex HRP) est remplacé par un Markowitz min variance
      (min_volatility) sur la fenêtre MOIS_MARKOWITZ, après le même filtre
      momentum (on ne garde que les actions gagnantes) qu'utilisait HRP.

    La logique de combinaison (regime_switch sur la fenêtre longue, moyenne
    pondérée par alpha) est identique à hybride().
    """
    rendements_reels_mensuels_OPTI = []
    poids_prec = np.zeros(len(assets))
    
    for i in range(MOIS_TEST):
        # 🟢 CRÉATION DE DEUX FENÊTRES DE TEMPS DISTINCTES
        start_top_k = changer_date(current_end, -MOIS_TOP_K)
        start_markowitz = changer_date(current_end, -MOIS_MARKOWITZ)
        
        data_train_top_k = data.loc[start_top_k:current_end]
        data_train_markowitz = data.loc[start_markowitz:current_end]
        
        poids_opt_markowitz_def = np.zeros(len(assets))
        poids_opt_markowitz_agr = np.zeros(len(assets))
        
        # --- 1. Volet AGRESSIF : Top K momentum (fenêtre MOIS_TOP_K) puis Markowitz max Sharpe, cap 20% ---
        if not data_train_top_k.empty and len(data_train_top_k) > 1:
            rendement_top_k = (data_train_top_k.iloc[-1] / data_train_top_k.iloc[0]) - 1
            top_actions = rendement_top_k.sort_values(ascending=False).head(K)
            top_actions = top_actions[top_actions > 0]
            
            # On ne lance Markowitz que s'il reste au moins 2 actions gagnantes
            if len(top_actions) > 1:
                data_train_top_k_filtre = data_train_top_k[top_actions.index]
                returns_train_agr = data_train_top_k_filtre.pct_change().dropna()
                if not returns_train_agr.empty:
                    try:
                        mu_agr = returns_train_agr.mean() * ANN_FACTOR
                        sigma_agr = returns_train_agr.cov() * ANN_FACTOR
                        ef_agr = pf.EfficientFrontier(mu_agr, sigma_agr, weight_bounds=(0, weight_limit))
                        raw_weights_agr = ef_agr.max_sharpe(risk_free_rate=RISK_FREE_RATE)
                        poids_opt_markowitz_agr = np.array([raw_weights_agr.get(ticker, 0) for ticker in assets])
                    except Exception as e:
                        warnings.warn(f"Optimiser failed at {current_end}: {e}")
            elif len(top_actions) == 1:
                # Une seule action gagnante : tout le poids du volet agressif va sur elle
                # (un solveur Markowitz à 1 seul actif n'a pas de sens, et le cap 20% par
                # actif ne peut pas s'appliquer à un portefeuille mono-actif).
                idx = assets.index(top_actions.index[0])
                poids_opt_markowitz_agr[idx] = 1.0
                        
        # --- 2. Volet DÉFENSIF : filtre momentum (fenêtre MOIS_MARKOWITZ) puis Markowitz min variance ---
        if not data_train_markowitz.empty and len(data_train_markowitz) > 1:
            rendement_def = (data_train_markowitz.iloc[-1] / data_train_markowitz.iloc[0]) - 1
            actions_gagnantes = rendement_def[rendement_def > 0].index.tolist()
            data_train_filtre = data_train_markowitz[actions_gagnantes]
            
            if not data_train_filtre.empty and len(data_train_filtre.columns) > 1:
                returns_train_def = data_train_filtre.pct_change().dropna()
                if not returns_train_def.empty:
                    try:
                        mu_def = returns_train_def.mean() * ANN_FACTOR
                        sigma_def = returns_train_def.cov() * ANN_FACTOR
                        ef_def = pf.EfficientFrontier(mu_def, sigma_def)
                        raw_weights_def = ef_def.min_volatility()
                        poids_opt_markowitz_def = np.array([raw_weights_def.get(ticker, 0) for ticker in assets])
                    except Exception as e:
                        warnings.warn(f"Optimiser failed at {current_end}: {e}")

        # --- D. Données de test ---
        test_start = current_end
        test_end = changer_date(current_end, 1)
        data_test = data.loc[test_start:test_end]
    
        # --- E. Calcul OOS ---
        if not data_test.empty:
            ret_oos_OPTI_agr = rendement_reel_oos(poids_opt_markowitz_agr, data_test)
            ret_oos_OPTI_def = rendement_reel_oos(poids_opt_markowitz_def, data_test)
        else:
            ret_oos_OPTI_agr = 0
            ret_oos_OPTI_def = 0
            
        # On utilise la fenêtre longue (Markowitz défensif) pour définir le régime de l'indice de marché
        alpha = regime_switch(start_markowitz, current_end, data_indice)
        
        # Moyenne pondérée : alpha = poids du volet défensif (élevé en tendance baissière).
        rendement_hybride_du_mois = (ret_oos_OPTI_agr * (1 - alpha)) + (ret_oos_OPTI_def * alpha)
        poids_hyb = (1 - alpha) * poids_opt_markowitz_agr + alpha * poids_opt_markowitz_def
        rendement_hybride_du_mois -= frais_rotation(poids_hyb, poids_prec)
        poids_prec = poids_hyb
        rendements_reels_mensuels_OPTI.append(rendement_hybride_du_mois)
        
        # --- F. Glissement ---
        current_end = changer_date(current_end, 1)
        
    return rendements_reels_mensuels_OPTI