# CLAUDE_LOCAL.md — Instructions pour Claude Code (PC local)

Ce fichier est destiné à Claude Code tournant sur le PC local Windows.
Le projet original tourne sur une instance AWS Ubuntu distante.

## Contexte du projet

PolyBot Crypto est un market-maker automatique sur Polymarket (options binaires BTC).
Il utilise la formule d'Avellaneda-Stoikov (AS-lite) centrée sur une fair value Black-Scholes.

**On est ici pour faire du backtesting offline uniquement.**
Aucune connexion réseau. Aucun ordre réel. Pas besoin de `.env`.

---

## Objectif principal

Comparer deux stratégies de market-making sur les données historiques :

| Stratégie | Fichier | Description |
|---|---|---|
| **AS-lite (baseline)** | `tools/backtest_comparison.py` | Heuristique Avellaneda-Stoikov centrée sur FV Black-Scholes |
| **RL PPO** | `tools/backtest_comparison.py` | Agent PPO entraîné sur les mêmes données |

Le modèle RL est dans `data/rl_models/ppo_mm_20260306_1623.zip`.

---

## Setup Python

```bash
# Python 3.11+ recommande
pip install stable-baselines3 pandas numpy matplotlib gymnasium
```

Pas besoin de : web3, py-clob-client, websockets, curl_cffi, httpx, eth_account.
Ces deps sont pour la prod (AWS), pas pour le backtest.

---

## Lancer le backtest comparatif

```bash
cd C:\perso\Polybot_AI_Lab

# Backtest standard (tous les jours disponibles)
python tools/backtest_comparison.py

# Sur les N derniers jours seulement
python tools/backtest_comparison.py --days 14

# Avec un modele RL specifique
python tools/backtest_comparison.py --model data/rl_models/ppo_mm_20260306_1623.zip --days 20
```

Le script charge les CSV dans `data/`, les sous-echantillonne a 2000 ticks/jour pour la RAM,
et rejoue tick par tick avec un modele de fill deterministe.

---

## Structure des fichiers importants

```
polybot_ai_lab/
├── config.py                        # Parametres centraux (MM_HALF_SPREAD_CENTS, etc.)
├── strategy/
│   ├── avellaneda_stoikov.py        # Formule AS-lite (calcul des quotes)
│   ├── pricing.py                   # Black-Scholes binary (fair value)
│   ├── microstructure.py            # VPIN, momentum, imbalance
│   └── fill_model.py                # Modele de fill pour simulation
├── tools/
│   ├── backtest_comparison.py       # SCRIPT PRINCIPAL — AS-lite vs RL PPO
│   ├── rl_ppo_train.py              # Re-entrainement RL (si besoin)
│   ├── rl_quick_test.py             # Test rapide du modele RL
│   └── build_rl_dataset.py          # Construction du dataset d'entrainement RL
└── data/
    ├── dataset_2026-02-*.csv        # Donnees historiques Fevrier
    ├── dataset_2026-03-*.csv        # Donnees historiques Mars
    └── rl_models/
        └── ppo_mm_20260306_1623.zip # Modele PPO entraine (stable-baselines3)
```

---

## Schema des CSV

Les fichiers `dataset_YYYY-MM-DD.csv` ont jusqu'a 40 colonnes selon la date.
Colonnes cles pour le backtest :

| Colonne | Type | Description |
|---|---|---|
| `best_bid` | float | Meilleur bid Polymarket (ex: 0.62) |
| `best_ask` | float | Meilleur ask Polymarket (ex: 0.65) |
| `fv_theoretical` | float | Fair value Black-Scholes (ex: 0.638) |
| `bid_vol_1` | float | Volume au meilleur bid |
| `ask_vol_1` | float | Volume au meilleur ask |
| `imbalance` | float | Desequilibre OB (bid_vol - ask_vol) / total |
| `hours_left` | float | Heures avant expiry du marche |
| `implied_vol` | float | Volatilite implicite (Kalman ou Deribit) |
| `momentum` | float | Signal momentum BTC court terme |

Multi-schema (certains fichiers anciens n'ont que 10 ou 22 colonnes) — toujours utiliser
`on_bad_lines='skip'` et `usecols=lambda c: c in USE_COLS` comme fait dans backtest_comparison.py.

---

## Resultats attendus du backtest

Le backtest affiche pour chaque jour et en cumule :

- **PnL realise** (USDC) — revenus du spread apres fills
- **Sharpe ratio** — PnL / volatilite journaliere
- **Fill rate** — % de ticks ou un ordre est touche
- **Inventory max** — exposition maximale (en shares)
- **Adverse selection** — perte moyenne quand le marche bouge contre nous apres fill

Resultats precedents (depuis MEMORY.md) :
- AS-lite heuristique : ~+44 USDC/jour (simule)
- RL PPO : ~+54 USDC/jour (simule, +22% vs heuristique)

Ces chiffres sont sur donnees in-sample. L'objectif est de valider l'ecart sur OOS.

---

## Taches a faire (dans l'ordre)

### 1. Verifier l'environnement
```bash
python -c "import stable_baselines3; import pandas; import numpy; print('OK')"
```

### 2. Lancer le backtest de base
```bash
python tools/backtest_comparison.py --days 20
```
Verifier que les deux strategies tournent sans erreur et affichent des resultats.

### 3. Analyse OOS (out-of-sample)
Le modele RL a ete entraine sur les donnees avant le 06/03/2026.
Tester sur les **7 derniers jours** (06-07/03) pour valider la generalisation :
```bash
python tools/backtest_comparison.py --days 7
```

### 4. Re-entrainement si necessaire
Si le RL est mauvais en OOS, relancer l'entrainement sur plus de donnees :
```bash
python tools/rl_ppo_train.py
```
Le script construit le dataset, entraine PPO, et sauvegarde un nouveau .zip.

### 5. Optimisation des hyperparametres AS-lite
Faire un grid search sur les params AS-lite pour trouver le meilleur baseline :
```bash
python tools/param_optimizer.py
```

---

## Notes techniques importantes

1. **CSV multi-schema** : Toujours `on_bad_lines='skip'`. Certains jours anciens ont 10 colonnes,
   les recents en ont 38-40. Le backtest gere ca automatiquement.

2. **RAM** : Le backtest sous-echantillonne a 2000 ticks/jour par defaut. Si le PC a
   suffisamment de RAM (>8GB), tu peux augmenter dans `backtest_comparison.py` :
   `step = max(1, len(df) // 5000)` pour plus de precision.

3. **Modele RL** : Charge via `stable_baselines3.PPO.load(path)`. L'environnement Gym
   doit correspondre exactement a celui utilise a l'entrainement (voir `rl_ppo_train.py`).

4. **Pas de `.env`** : Aucune cle privee necessaire. Le backtest est 100% offline.

5. **Chemin de travail** : Toujours lancer depuis la racine du projet :
   `cd C:\perso\Polybot_AI_Lab && python tools/backtest_comparison.py`

---

## En cas de bug

- `ModuleNotFoundError: stable_baselines3` → `pip install stable-baselines3 gymnasium`
- `KeyError: 'fv_theoretical'` → CSV trop ancien (pre-Feb 8), ignorer ces fichiers avec `--days 25`
- `MemoryError` → Reduire `step` dans `load_days()` pour moins de ticks par jour
- Le modele RL predit des actions hors range → verifier que `RL_SPREAD_MAP` correspond a l'entrainement

---

## Contact / contexte serveur

Le bot live tourne en parallele sur AWS (screen `polybot`).
Ne pas modifier les fichiers de prod a distance depuis ici.
Les CSV sur le PC local sont une copie figee au 07/03/2026.
