#!/usr/bin/env python3
"""
Optimiseur Bayésien de paramètres MM — Optuna TPE + Walk-Forward Validation.

Pourquoi Bayésien plutôt que Grid Search ?
───────────────────────────────────────────────────────────────────────────────
Grid Search : explore 5×5×4 = 100 combinaisons uniformément → inefficace car
  la surface de score est lisse et creuse, les optimums sont rares.

Optimisation Bayésienne (Bayesian Optimization) :
  1. Surrogate model : modélise f(gamma, max_inv, spread) avec un modèle
     probabiliste (ici TPE — Tree-structured Parzen Estimators).
  2. Acquisition function : sélectionne le prochain point à évaluer en
     maximisant l'Expected Improvement (EI = E[max(f(x) - best, 0)]).
  3. Update : après chaque évaluation, le surrogate est mis à jour.

  → Trouve de meilleurs paramètres en ~50 évaluations au lieu de 100+,
    car il concentre les évaluations là où la surface semble prometteuse
    tout en explorant suffisamment pour ne pas rater l'optimum global.

TPE vs GP (Gaussian Process) :
  - GP : modèle global, puissant mais coûteux (O(n³) par fit), calibrage délicat.
  - TPE : modèle non-paramétrique basé sur histogrammes de densité → O(n log n),
    gère naturellement les variables discrètes (max_inv, spread_ticks = entiers),
    robuste au bruit de mesure. Recommandé pour <10 paramètres.

Refs: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
      Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"

Usage:
    python3 tools/param_optimizer.py           # analyse seule
    python3 tools/param_optimizer.py --apply   # patche config.py si fiable
    python3 tools/param_optimizer.py --days 25 # N derniers jours
    python3 tools/param_optimizer.py --n_trials 80 # nombre d'évaluations
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import re
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silencieux sauf erreurs

# ── CONSTANTES SIMULATION ─────────────────────────────────────────────────────

DATA_DIR        = "data"
MIN_TICKS_RAW   = 2000
RESAMPLE_S      = 60
MIN_HOURS_LEFT  = 0.5
MIN_SENSITIVITY = 0.04
QUOTE_SIZE      = 5.0
TICK_STD        = 0.01
SIGMA_OPT       = 0.025

OOS_DAYS        = 5
MIN_OOS_SCORE   = 0.0    # Score OOS minimum pour patcher config.py

# Score weights
W_CASH    = 1.0
W_MAX_INV = 0.10
W_BALANCE = 0.02

# Espace de recherche Bayésien
SEARCH_SPACE = {
    "gamma"       : (0.05, 2.5),    # continu
    "max_inv"     : (5, 50),        # entier
    "spread_ticks": (1, 8),         # entier
}

# ── CHARGEMENT DONNÉES ────────────────────────────────────────────────────────

def load_day(filepath):
    """Charge et prépare un CSV journalier. Retourne DataFrame ou None."""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
    except Exception:
        return None

    needed = {'timestamp', 'best_bid', 'best_ask', 'fv_theoretical',
              'hours_left', 'market_strike'}
    if not needed.issubset(df.columns):
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'fv_theoretical', 'best_bid', 'best_ask'])
    df = df[(df['best_bid'] > 0) & (df['best_ask'] > 0)]
    df = df[(df['fv_theoretical'] > 0.01) & (df['fv_theoretical'] < 0.99)]
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df if len(df) >= MIN_TICKS_RAW else None


def resample_by_strike(df):
    """Agrège à RESAMPLE_S secondes par strike. Retourne dict {strike: np.ndarray}."""
    df = df.copy().set_index('timestamp')
    rule = f"{RESAMPLE_S}s"
    MIN_ROWS = 50
    results = {}

    for strike, gdf in df.groupby('market_strike'):
        r = gdf.resample(rule).agg({
            'best_bid'       : 'last',
            'best_ask'       : 'last',
            'fv_theoretical' : 'mean',
            'hours_left'     : 'last',
        }).dropna()

        if len(r) < MIN_ROWS:
            continue

        arr = r[['fv_theoretical', 'hours_left', 'best_bid', 'best_ask']].values.astype(np.float64)
        nxt_bid = np.roll(arr[:, 2], -1)
        nxt_ask = np.roll(arr[:, 3], -1)
        data = np.column_stack([arr, nxt_bid, nxt_ask])
        results[strike] = data[:-1]

    return results

# ── SIMULATION A-S (inchangée — cœur du backtest) ────────────────────────────

def _tick_round_arr(prices):
    result = np.round(prices / TICK_STD) * TICK_STD
    extreme = (prices < 0.10) | (prices > 0.90)
    result[extreme] = np.round(prices[extreme] / 0.001) * 0.001
    return np.round(result, 4)


def simulate_strike(data, gamma, max_inv, spread_ticks):
    """
    Simulation vectorisée Avellaneda-Stoikov sur un array numpy [N, 6].
    Colonnes : fv, hours, mkt_bid, mkt_ask, nxt_bid, nxt_ask.
    Retourne dict de métriques.
    """
    fv       = data[:, 0]
    hours    = data[:, 1]
    mkt_bid  = data[:, 2]
    mkt_ask  = data[:, 3]
    nxt_bid  = data[:, 4]
    nxt_ask  = data[:, 5]
    n        = len(data)

    sensitivity = fv * (1 - fv)
    valid = (hours >= MIN_HOURS_LEFT) & (sensitivity >= MIN_SENSITIVITY) & \
            (mkt_bid > 0) & (mkt_ask > 0) & (mkt_bid < mkt_ask)

    initial_hours = hours[0] if n > 0 else 17.0
    inventory = 0.0
    cash      = 0.0
    buy_fills = 0
    sell_fills = 0
    max_abs_inv = 0.0

    for i in range(n):
        if not valid[i]:
            continue

        T_norm  = hours[i] / max(initial_hours, 1.0)
        reserve = fv[i] - gamma * (SIGMA_OPT ** 2) * inventory * T_norm
        half_sp = max(
            spread_ticks * TICK_STD / 2,
            gamma * (SIGMA_OPT ** 2) * T_norm / 2,
        )

        bid = _tick_round_arr(np.array([reserve - half_sp]))[0]
        ask = _tick_round_arr(np.array([reserve + half_sp]))[0]
        bid = max(TICK_STD, min(bid, 0.98))
        ask = min(1 - TICK_STD, max(ask, 0.02))

        if bid >= ask:
            continue

        nb = nxt_bid[i]
        na = nxt_ask[i]

        if bid > mkt_bid[i] and na <= bid and inventory < max_inv:
            inventory += QUOTE_SIZE
            cash      -= bid * QUOTE_SIZE
            buy_fills += 1
        elif ask < mkt_ask[i] and nb >= ask and inventory > -max_inv:
            inventory -= QUOTE_SIZE
            cash      += ask * QUOTE_SIZE
            sell_fills += 1

        abs_inv = abs(inventory)
        if abs_inv > max_abs_inv:
            max_abs_inv = abs_inv

    last_fv = float(fv[-1]) if n > 0 else 0.5
    cash   += inventory * last_fv
    matched = min(buy_fills, sell_fills)
    score   = W_CASH * cash - W_MAX_INV * max_abs_inv + W_BALANCE * matched

    return {
        'score'    : score,
        'cash'     : cash,
        'n_buy'    : buy_fills,
        'n_sell'   : sell_fills,
        'matched'  : matched,
        'max_inv'  : max_abs_inv,
        'final_inv': inventory,
    }

# ── OBJECTIF POUR OPTUNA ──────────────────────────────────────────────────────

def make_objective(day_datasets):
    """
    Retourne la fonction objectif pour Optuna.
    Optuna minimise par défaut → on retourne -score (car on veut MAXIMISER le score).
    """
    def objective(trial):
        gamma        = trial.suggest_float("gamma", *SEARCH_SPACE["gamma"])
        max_inv      = trial.suggest_int("max_inv", *SEARCH_SPACE["max_inv"])
        spread_ticks = trial.suggest_int("spread_ticks", *SEARCH_SPACE["spread_ticks"])

        scores = []
        for day_data in day_datasets:
            for data_arr in day_data.values():
                m = simulate_strike(data_arr, gamma, max_inv, spread_ticks)
                scores.append(m['score'])

        return -np.mean(scores) if scores else 99.0  # négatif car optuna minimise

    return objective


def run_bayesian_search(day_datasets, n_trials, verbose=False):
    """
    Optimisation Bayésienne via Optuna (TPE sampler).
    Retourne DataFrame des résultats trié par score décroissant.
    """
    study = optuna.create_study(
        direction="minimize",   # on minimise -score
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=max(10, n_trials // 5),  # phase exploration initiale
            multivariate=True,    # modélise les corrélations entre params
            seed=42,
        ),
    )
    study.optimize(
        make_objective(day_datasets),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({
                'gamma'       : t.params['gamma'],
                'max_inv'     : t.params['max_inv'],
                'spread_ticks': t.params['spread_ticks'],
                'score'       : -t.value,
            })

    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    return df, study


# ── WALK-FORWARD ──────────────────────────────────────────────────────────────

def walk_forward(all_datasets, n_trials, verbose=False):
    """
    IS = tout sauf les OOS_DAYS derniers jours.
    OOS = les OOS_DAYS derniers jours.
    1. Optimisation Bayésienne sur IS.
    2. Validation OOS des top-5 IS sur les jours réservés.
    Retourne (best_params_dict, oos_score_float, df_results).
    """
    if len(all_datasets) <= OOS_DAYS:
        print(f"  Peu de jours ({len(all_datasets)}) → entraînement sur tout le dataset")
        df, study = run_bayesian_search(all_datasets, n_trials, verbose)
        best = df.iloc[0]
        return {
            'gamma'       : round(best['gamma'], 4),
            'max_inv'     : int(best['max_inv']),
            'spread_ticks': int(best['spread_ticks']),
        }, float(best['score']), df

    is_data  = all_datasets[:-OOS_DAYS]
    oos_data = all_datasets[-OOS_DAYS:]

    print(f"  IS : {len(is_data)} jours | OOS : {len(oos_data)} jours")
    print(f"  Lancement optimisation Bayésienne ({n_trials} évaluations)...")

    df, study = run_bayesian_search(is_data, n_trials, verbose)

    # Valider les top-5 IS sur OOS
    best_oos_score = -np.inf
    best_params    = None

    print("  Validation OOS des top-5 paramètres IS...")
    for _, row in df.head(5).iterrows():
        oos_scores = []
        for day_data in oos_data:
            for data_arr in day_data.values():
                m = simulate_strike(data_arr, row['gamma'], row['max_inv'], row['spread_ticks'])
                oos_scores.append(m['score'])

        oos_mean = np.mean(oos_scores) if oos_scores else -99.0
        gamma_r  = round(row['gamma'], 4)
        print(f"    gamma={gamma_r:.2f} max_inv={row['max_inv']:.0f} "
              f"spread={row['spread_ticks']:.0f} → OOS={oos_mean:.3f}")

        if oos_mean > best_oos_score:
            best_oos_score = oos_mean
            best_params = {
                'gamma'       : round(row['gamma'], 4),
                'max_inv'     : int(row['max_inv']),
                'spread_ticks': int(row['spread_ticks']),
            }

    return best_params, best_oos_score, df

# ── PATCH CONFIG ──────────────────────────────────────────────────────────────

def patch_config(params, config_path="config.py"):
    shutil.copy(config_path, config_path + ".bak")
    with open(config_path) as f:
        content = f.read()

    for key, val in [
        ('MM_GAMMA',            params['gamma']),
        ('MM_MAX_INVENTORY',    params['max_inv']),
        ('MM_MIN_SPREAD_TICKS', params['spread_ticks']),
    ]:
        content = re.sub(
            rf'^({re.escape(key)}\s*=\s*)[\d.]+',
            rf'\g<1>{val}',
            content,
            flags=re.MULTILINE,
        )

    with open(config_path, 'w') as f:
        f.write(content)

    print(f"  config.py patché → gamma={params['gamma']} "
          f"max_inv={params['max_inv']} spread_ticks={params['spread_ticks']}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply',    action='store_true',
                    help='Patcher config.py si score OOS suffisant')
    ap.add_argument('--days',     type=int, default=25,
                    help='Nombre de jours à utiliser (défaut: 25)')
    ap.add_argument('--n_trials', type=int, default=60,
                    help='Nombre d\'évaluations Bayésiennes (défaut: 60)')
    ap.add_argument('--verbose',  action='store_true',
                    help='Affiche la barre de progression')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f][-args.days:]

    print(f"\n{'='*60}")
    print(f"  OPTIMISEUR BAYESIEN (TPE) — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")
    print(f"Chargement de {len(files)} fichiers CSV...")

    import time
    t0 = time.time()

    all_datasets = []
    for f in files:
        df = load_day(f)
        if df is None:
            continue
        day_data = resample_by_strike(df)
        if day_data:
            all_datasets.append(day_data)
            total_rows = sum(len(v) for v in day_data.values())
            strikes    = [int(s) for s in day_data.keys()]
            print(f"  {os.path.basename(f)}: {total_rows} lignes | strikes: {strikes}")

    print(f"\n{len(all_datasets)} journées valides chargées en {time.time()-t0:.1f}s\n")

    if len(all_datasets) < 3:
        print("ERREUR : minimum 3 jours de données requis.")
        sys.exit(1)

    t1 = time.time()
    best_params, oos_score, df_results = walk_forward(
        all_datasets, args.n_trials, args.verbose
    )
    elapsed = time.time() - t1

    # Affichage résultats
    print(f"\n{'─'*60}")
    print(f"  TOP 10 — In-Sample (calculé en {elapsed:.1f}s, {args.n_trials} évaluations)")
    print(f"{'─'*60}")
    cols = ['gamma', 'max_inv', 'spread_ticks', 'score']
    top10 = df_results.head(10).copy()
    top10['gamma'] = top10['gamma'].round(3)
    print(top10[cols].to_string(index=False, float_format='{:.3f}'.format))

    print(f"\n{'─'*60}")
    print(f"  RECOMMANDATION BAYESIENNE (score OOS: {oos_score:.3f})")
    print(f"{'─'*60}")
    print(f"  gamma         = {best_params['gamma']}")
    print(f"  max_inventory = {best_params['max_inv']}")
    print(f"  spread_ticks  = {best_params['spread_ticks']}")

    # Comparaison avec config actuelle
    try:
        import importlib
        import config as cfg_mod
        importlib.reload(cfg_mod)
        current = {
            'gamma'       : cfg_mod.MM_GAMMA,
            'max_inv'     : cfg_mod.MM_MAX_INVENTORY,
            'spread_ticks': cfg_mod.MM_MIN_SPREAD_TICKS,
        }
        print(f"\n  Config actuelle : gamma={current['gamma']} "
              f"max_inv={current['max_inv']} spread_ticks={current['spread_ticks']}")
        delta = {k: f"{current.get(k)} → {v}" for k, v in best_params.items()
                 if v != current.get(k)}
        if delta:
            print(f"  Changements proposés : {delta}")
        else:
            print(f"  Params déjà optimaux selon les données disponibles.")
    except Exception:
        pass

    # Application si demandé
    applied = False
    if args.apply:
        if oos_score >= MIN_OOS_SCORE:
            patch_config(best_params)
            applied = True
        else:
            print(f"\n  Score OOS {oos_score:.3f} < seuil {MIN_OOS_SCORE} "
                  f"→ params NON appliqués (signal insuffisant out-of-sample)")
    else:
        print(f"\n  Pour appliquer : python3 tools/param_optimizer.py --apply")

    # JSON de résultats (lu par cron + daily_analyst)
    result = {
        'timestamp'  : datetime.now().isoformat(),
        'best_params': best_params,
        'oos_score'  : oos_score,
        'n_days'     : len(all_datasets),
        'n_trials'   : args.n_trials,
        'method'     : 'bayesian_tpe',
        'applied'    : applied,
    }
    out = os.path.join(DATA_DIR, 'optimizer_result.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Résultat → {out}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
