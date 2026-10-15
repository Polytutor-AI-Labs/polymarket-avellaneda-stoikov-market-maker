# CLAUDE.md

## Contexte

Ce dossier est la partie **analyse & backtesting** du projet PolyBot Crypto, un market maker pour Polymarket (options binaires BTC).
Le code principal tournait sur une VM Ubuntu distante. Ici c'est la copie locale (Windows) pour faire tourner les analyses lourdes sans crasher la VM.

L'utilisateur communique en **francais**.

## Structure

```
analysis/
  feature_analysis.ipynb   # Analyse des features (imbalance, spread, volatilite)
  backtesting.ipynb         # Grid search: gamma, spread, max_inventory
  mm_backtest.py            # Moteur de backtest market-making (le plus gros fichier)
  utils.py                  # Fonctions communes: load_datasets(), metriques
  results/                  # Resultats sauvegardes
data/
  dataset_YYYY-MM-DD.csv    # Donnees tick-level (~10 ticks/sec, 30MB/jour)
  trades_journal.csv        # Journal des trades executes
  best_markets.json         # Marche cible actuel (strike, expiry, token IDs)
venv/                       # Environnement Python local
```

## Donnees CSV

- **Attention** : plusieurs schemas (10, 22, 27 colonnes) selon les dates
- Utiliser `on_bad_lines='skip'` pour les lignes corrompues
- `utils.py` a un parametre `min_cols` pour filtrer les vieux schemas
- Colonnes recentes : timestamp, btc_price, best_bid, best_ask, bid_vol_1, ask_vol_1, imbalance, fv_theoretical, edge, signal, spread, gamma, inventory, theo_bid, theo_ask, ...

## Chemins Windows

Tous les chemins doivent utiliser des raw strings :
```python
DATA_PATH = r"C:\perso\Polybot_AI_Lab\data"
TRADES_PATH = r"C:\perso\Polybot_AI_Lab\data\trades_journal.csv"
```

## Commandes

```powershell
# Activer venv
.\venv\Scripts\Activate.ps1

# Installer dependances
pip install -r analysis\requirements_local.txt

# Lancer Jupyter
$env:PYTHONUTF8 = "1"
jupyter notebook analysis/

# Ou lancer un backtest directement
python analysis\mm_backtest.py
```

## Objectif

Trouver les parametres optimaux **(gamma, spread, max_inventory)** via backtesting pour ameliorer le P&L du bot.
- Strategie : Avellaneda-Stoikov + Black-Scholes fair value
- Metriques prioritaires : **Sharpe ratio > P&L brut** (eviter l'overfitting)
- Toujours valider sur donnees out-of-sample (split chronologique)

## Environnement

- Windows, Python 3.12, `PYTHONUTF8=1` obligatoire (emojis dans les logs)
- Toujours restart le kernel Jupyter apres modification de utils.py
- La VM source est en ligne (instance Ubuntu AWS, accès SSH) si besoin de recup de nouvelles donnees
