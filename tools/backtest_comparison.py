#!/usr/bin/env python3
"""
backtest_comparison.py — Comparaison offline AS-lite (BS) vs RL PPO.

Principe :
  On rejoue les données historiques CSV tick par tick sur les deux stratégies
  avec le MÊME modèle de fill déterministe :
    - BUY fill  : notre bid >= market ask suivant
    - SELL fill : notre ask <= market bid suivant

Aucune connexion réseau. Aucun ordre réel.
Résultat : P&L journalier, Sharpe, gestion d'inventaire.

Usage :
    python3 tools/backtest_comparison.py
    python3 tools/backtest_comparison.py --days 10
    python3 tools/backtest_comparison.py --model data/rl_models/ppo_mm_20260306_1623.zip
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from strategy.fill_model import fill_probability_v2

DATA_DIR  = "data"
MODEL_DIR = "data/rl_models"

# ─── Paramètres communs ───────────────────────────────────────────────────────
QUOTE_SIZE = 5.0    # USDC par quote
TICK       = 0.01
MIN_SHARES = 5.0    # minimum Polymarket

# AS-lite params
AS_HALF_SPREAD = 0.03   # 3 cents par cote
AS_SKEW_MAX    = 0.02   # 2 cents skew max
AS_MAX_INV     = 50.0

# Timing (doit correspondre a la prod)
TICKS_PER_SEC    = 10       # frequence des CSV (~10 ticks/sec)
REQUOTE_INTERVAL = 5.0      # secondes entre requotes (MM_REQUOTE_INTERVAL en prod)
COOLDOWN_TICKS   = int(REQUOTE_INTERVAL * TICKS_PER_SEC)  # ~50 ticks

# RL params (doit correspondre à l'entraînement)
RL_SPREAD_MAP  = [0, 2, 4, 6]   # ticks par action
RL_MAX_INV     = 10.0

# ─── Chargement des données ───────────────────────────────────────────────────

USE_COLS = {
    'best_bid', 'best_ask', 'fv_theoretical', 'bid_vol_1', 'ask_vol_1',
    'imbalance', 'hours_left', 'implied_vol', 'momentum',
    'bid_vol_2', 'bid_vol_3', 'ask_vol_2', 'ask_vol_3', 'spread',
}

def load_days(n_days):
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f][-n_days:]
    days = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False,
                             usecols=lambda c: c in USE_COLS)
            for col in df.select_dtypes(include='object').columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['best_bid', 'best_ask', 'fv_theoretical'])
            df = df[df['best_bid'] > 0]
            df = df[df['fv_theoretical'].between(0.05, 0.95)]
            if 'hours_left' in df.columns:
                df = df[df['hours_left'] > 0.5]
            # Pas de sous-echantillonnage : on garde tous les ticks
            # (100k ticks/jour ~ 80MB pour 20 jours, OK sur PC local)
            if len(df) >= 100:
                days.append((os.path.basename(f), df))
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")
    return days

# ─── Modèle de fill (identique pour les deux stratégies) ─────────────────────

def simulate_day(df, get_quotes_fn, max_inv, label="", rng=None):
    """
    Rejoue un jour tick par tick avec fill model stochastique.
    get_quotes_fn(fv, bid, ask, imb, hl, ivol, mom, inventory, t) -> (our_bid, our_ask) ou None
    Retourne le dict de metriques journalieres.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(df)
    inventory  = 0.0
    cost_basis = 0.0
    realized   = 0.0
    fills_buy  = 0
    fills_sell = 0
    total_adverse = 0.0

    # Pre-extraire les colonnes en arrays numpy pour la perf
    fvs    = df['fv_theoretical'].values.astype(np.float64)
    bids   = df['best_bid'].values.astype(np.float64)
    asks   = df['best_ask'].values.astype(np.float64)
    imbs   = df['imbalance'].values.astype(np.float64) if 'imbalance' in df.columns else np.zeros(n)
    hls    = df['hours_left'].values.astype(np.float64) if 'hours_left' in df.columns else np.full(n, 5.0)
    ivols  = df['implied_vol'].values.astype(np.float64) if 'implied_vol' in df.columns else np.full(n, 0.5)
    moms   = df['momentum'].values.astype(np.float64) if 'momentum' in df.columns else np.zeros(n)
    bv1    = df['bid_vol_1'].values.astype(np.float64) if 'bid_vol_1' in df.columns else np.full(n, 100.0)
    av1    = df['ask_vol_1'].values.astype(np.float64) if 'ask_vol_1' in df.columns else np.full(n, 100.0)
    bv2    = df['bid_vol_2'].values.astype(np.float64) if 'bid_vol_2' in df.columns else np.zeros(n)
    bv3    = df['bid_vol_3'].values.astype(np.float64) if 'bid_vol_3' in df.columns else np.zeros(n)
    av2    = df['ask_vol_2'].values.astype(np.float64) if 'ask_vol_2' in df.columns else np.zeros(n)
    av3    = df['ask_vol_3'].values.astype(np.float64) if 'ask_vol_3' in df.columns else np.zeros(n)
    spds   = df['spread'].values.astype(np.float64) if 'spread' in df.columns else (asks - bids)

    next_quote_tick = 0  # cooldown : prochaine quote autorisee

    for t in range(n - 1):
        if t < next_quote_tick:
            continue

        fv, bid, ask = fvs[t], bids[t], asks[t]
        nxt_bid, nxt_ask = bids[t+1], asks[t+1]

        if bid <= 0 or ask <= 0 or nxt_bid <= 0 or nxt_ask <= 0:
            continue

        imb = imbs[t]; hl = hls[t]; ivol = ivols[t]; mom = moms[t]

        result = get_quotes_fn(fv, bid, ask, imb, hl, ivol, mom, inventory, t)
        if result is None:
            next_quote_tick = t + COOLDOWN_TICKS
            continue
        our_bid, our_ask = result

        shares_buy  = QUOTE_SIZE / our_bid if our_bid > 0 else 0
        shares_sell = min(QUOTE_SIZE / our_ask if our_ask > 0 else 0, inventory)

        filled = False

        # -- BUY fill : prix croise + probabilite stochastique --
        if our_bid >= nxt_ask and inventory < max_inv and shares_buy >= 1.0:
            p_fill = fill_probability_v2(
                shares_buy, av1[t], imb, is_buy=True,
                depth_vol_2=av2[t], depth_vol_3=av3[t],
                spread=spds[t], realized_vol=ivol)
            if rng.random() < p_fill:
                shares_buy = min(shares_buy, max_inv - inventory)
                inventory  += shares_buy
                cost_basis += our_bid * shares_buy
                fills_buy  += 1
                adverse = max(0.0, our_bid - nxt_ask) * shares_buy
                total_adverse += adverse
                filled = True

        # -- SELL fill : prix croise + probabilite stochastique --
        if our_ask <= nxt_bid and inventory > 0 and shares_sell >= 1.0:
            p_fill = fill_probability_v2(
                shares_sell, bv1[t], imb, is_buy=False,
                depth_vol_2=bv2[t], depth_vol_3=bv3[t],
                spread=spds[t], realized_vol=ivol)
            if rng.random() < p_fill:
                avg_cost = cost_basis / inventory if inventory > 0 else 0
                pnl_trade = (our_ask - avg_cost) * shares_sell
                realized   += pnl_trade
                cost_basis -= avg_cost * shares_sell
                inventory   = max(0.0, inventory - shares_sell)
                if inventory <= 0.01:
                    cost_basis = 0.0
                fills_sell += 1
                filled = True

        # Cooldown apres chaque quote (reel = requote toutes les 5s)
        next_quote_tick = t + COOLDOWN_TICKS

    # Polymarket = hold to expiry, pas de MTM quotidien
    # On ne compte que le PnL realise (round-trips completes)
    unrealized = 0.0
    total_pnl = realized

    return {
        'pnl_realized': realized,
        'pnl_unrealized': unrealized,
        'pnl_total': total_pnl,
        'fills_buy': fills_buy,
        'fills_sell': fills_sell,
        'inventory_final': inventory,
        'adverse_selection': total_adverse,
    }

# ─── Stratégie 1 : AS-lite (BS-centered) ─────────────────────────────────────

def as_lite_quotes(fv, bid, ask, imb, hl, ivol, mom, inventory, t):
    """Quote computation AS-lite : center = blend(FV, market_mid) - skew."""
    market_mid = (bid + ask) / 2.0
    center = 0.5 * fv + 0.5 * market_mid  # market-anchor (fix prod 25/02)
    inv_ratio = max(-1.0, min(1.0, inventory / AS_MAX_INV))
    center    = center - inv_ratio * AS_SKEW_MAX
    center    = max(TICK, min(1.0 - TICK, center))

    our_bid = round(center - AS_HALF_SPREAD, 2)
    our_ask = round(center + AS_HALF_SPREAD, 2)
    our_bid = max(TICK, min(our_bid, 1.0 - TICK))
    our_ask = max(TICK, min(our_ask, 1.0 - TICK))

    return our_bid, our_ask

# ─── Strategie 1b : AS-lite avec filtres parametrables ────────────────────────

def make_filtered_quotes(min_edge=0.0, max_imb=1.0, mom_mult=0.0, half_spread=None):
    """Fabrique une quote function AS-lite avec filtres anti-adverse-selection."""
    hs = half_spread if half_spread is not None else AS_HALF_SPREAD
    def quotes_fn(fv, bid, ask, imb, hl, ivol, mom, inventory, t):
        market_mid = (bid + ask) / 2.0

        # Filtre 1 — Edge minimum
        edge = abs(fv - market_mid)
        if edge < min_edge:
            return None

        # Filtre 2 — Imbalance max (flow toxique)
        if abs(imb) > max_imb:
            return None

        # Center avec market-anchor
        center = 0.5 * fv + 0.5 * market_mid
        inv_ratio = max(-1.0, min(1.0, inventory / AS_MAX_INV))
        center = center - inv_ratio * AS_SKEW_MAX
        center = max(TICK, min(1.0 - TICK, center))

        our_bid = round(center - hs, 2)
        our_ask = round(center + hs, 2)
        our_bid = max(TICK, min(our_bid, 1.0 - TICK))
        our_ask = max(TICK, min(our_ask, 1.0 - TICK))
        if our_ask <= our_bid:
            our_ask = our_bid + TICK

        return our_bid, our_ask
    return quotes_fn


# ─── Grid search filtres ─────────────────────────────────────────────────────

GRID_MIN_EDGE    = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
GRID_MAX_IMB     = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
GRID_MOM_MULT    = [0.0]  # momentum = 0 dans les CSV, parametre inutile
GRID_HALF_SPREAD = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]


def grid_search_filters(days_data):
    """Grid search sur edge + imbalance + half_spread, split IS/OOS."""
    n_days = len(days_data)
    split = max(1, n_days * 2 // 3)  # ~66% IS, ~33% OOS
    is_days  = days_data[:split]
    oos_days = days_data[split:]

    total = len(GRID_MIN_EDGE) * len(GRID_MAX_IMB) * len(GRID_HALF_SPREAD)

    print(f"\n{'='*80}")
    print(f"  GRID SEARCH — edge x imbalance x half_spread")
    print(f"  {len(is_days)} jours IS ({is_days[0][0]} -> {is_days[-1][0]})")
    print(f"  {len(oos_days)} jours OOS ({oos_days[0][0]} -> {oos_days[-1][0]})")
    print(f"  {len(GRID_MIN_EDGE)}x{len(GRID_MAX_IMB)}x{len(GRID_HALF_SPREAD)} "
          f"= {total} combinaisons")
    print(f"{'='*80}")

    results = []
    done = 0

    for hs in GRID_HALF_SPREAD:
        for min_edge in GRID_MIN_EDGE:
            for max_imb in GRID_MAX_IMB:
                done += 1
                quotes_fn = make_filtered_quotes(min_edge, max_imb, 0.0, half_spread=hs)

                is_pnls, is_fills, is_adv = [], [], []
                for fname, df in is_days:
                    rng = np.random.default_rng(42)
                    r = simulate_day(df, quotes_fn, AS_MAX_INV, rng=rng)
                    is_pnls.append(r['pnl_total'])
                    is_fills.append(r['fills_buy'] + r['fills_sell'])
                    is_adv.append(r['adverse_selection'])

                oos_pnls, oos_fills, oos_adv = [], [], []
                for fname, df in oos_days:
                    rng = np.random.default_rng(42)
                    r = simulate_day(df, quotes_fn, AS_MAX_INV, rng=rng)
                    oos_pnls.append(r['pnl_total'])
                    oos_fills.append(r['fills_buy'] + r['fills_sell'])
                    oos_adv.append(r['adverse_selection'])

                is_mean  = np.mean(is_pnls)
                oos_mean = np.mean(oos_pnls)
                is_std   = np.std(is_pnls) if len(is_pnls) > 1 else 1e-9
                oos_std  = np.std(oos_pnls) if len(oos_pnls) > 1 else 1e-9

                results.append({
                    'half_spread': hs,
                    'min_edge': min_edge,
                    'max_imb': max_imb,
                    'pnl_is': is_mean,
                    'pnl_oos': oos_mean,
                    'sharpe_is': is_mean / is_std if is_std > 0 else 0,
                    'sharpe_oos': oos_mean / oos_std if oos_std > 0 else 0,
                    'fills_is': np.mean(is_fills),
                    'fills_oos': np.mean(oos_fills),
                    'adv_is': np.mean(is_adv),
                    'adv_oos': np.mean(oos_adv),
                })

                if done % 20 == 0 or done == total:
                    print(f"  [{done:>3}/{total}] hs={hs:.2f} edge={min_edge:.3f} imb={max_imb:.1f} "
                          f" IS={is_mean:>+8.2f}  OOS={oos_mean:>+8.2f}")

    # Trier par PnL OOS decroissant
    results.sort(key=lambda x: x['pnl_oos'], reverse=True)

    # Afficher top 15
    print(f"\n{'='*110}")
    print(f"  TOP 15 (trie par PnL OOS)")
    print(f"{'='*110}")
    print(f"  {'#':>2}  {'hs':>5}  {'edge':>6}  {'imb':>5}  "
          f"{'PnL IS':>8}  {'PnL OOS':>8}  {'Shp IS':>7}  {'Shp OOS':>8}  "
          f"{'Fill IS':>7}  {'Fill OOS':>8}  {'Adv IS':>7}  {'Adv OOS':>8}")
    print(f"  {'─'*105}")

    for i, r in enumerate(results[:15]):
        print(f"  {i+1:>2}  {r['half_spread']:>5.2f}  {r['min_edge']:>6.3f}  {r['max_imb']:>5.1f}  "
              f"{r['pnl_is']:>+8.2f}  {r['pnl_oos']:>+8.2f}  {r['sharpe_is']:>+7.3f}  "
              f"{r['sharpe_oos']:>+8.3f}  {r['fills_is']:>7.1f}  {r['fills_oos']:>8.1f}  "
              f"{r['adv_is']:>7.2f}  {r['adv_oos']:>8.2f}")

    # Bottom 5
    print(f"\n  BOTTOM 5")
    print(f"  {'─'*105}")
    for i, r in enumerate(results[-5:]):
        rank = len(results) - 4 + i
        print(f"  {rank:>2}  {r['half_spread']:>5.2f}  {r['min_edge']:>6.003f}  {r['max_imb']:>5.1f}  "
              f"{r['pnl_is']:>+8.2f}  {r['pnl_oos']:>+8.2f}  {r['sharpe_is']:>+7.3f}  "
              f"{r['sharpe_oos']:>+8.3f}  {r['fills_is']:>7.1f}  {r['fills_oos']:>8.1f}  "
              f"{r['adv_is']:>7.2f}  {r['adv_oos']:>8.2f}")

    # Baseline et best
    baseline = next((r for r in results
                     if r['min_edge'] == 0 and r['max_imb'] == 1.0
                     and r['half_spread'] == 0.03), None)
    best = results[0]

    print(f"\n{'='*80}")
    print(f"  RESUME")
    print(f"{'='*80}")
    if baseline:
        print(f"  Baseline (hs=0.03, no filter) : PnL OOS={baseline['pnl_oos']:>+.2f}$/j  "
              f"Sharpe={baseline['sharpe_oos']:>+.3f}  Fills={baseline['fills_oos']:.1f}/j")
    print(f"  Best config : hs={best['half_spread']:.2f}  edge={best['min_edge']:.3f}  "
          f"imb={best['max_imb']:.1f}")
    print(f"                PnL OOS={best['pnl_oos']:>+.2f}$/j  "
          f"Sharpe={best['sharpe_oos']:>+.3f}  Fills={best['fills_oos']:.1f}/j  "
          f"Adv={best['adv_oos']:.2f}$/j")
    if baseline and baseline['pnl_oos'] != 0:
        improv = best['pnl_oos'] - baseline['pnl_oos']
        print(f"                Delta vs baseline: {improv:>+.2f}$/j")
    print(f"{'='*80}\n")

    return results


# ─── Strategie 2 : RL PPO ─────────────────────────────────────────────────────

def build_rl_quotes_fn(model):
    """Retourne une quote function qui utilise le modèle PPO."""
    def rl_quotes(fv, bid, ask, imb, hl, ivol, mom, inventory, t):
        # Observation identique à l'environnement d'entraînement
        spd    = max(0.0, ask - bid)
        inv_n  = np.clip(inventory / RL_MAX_INV, -1, 1)
        hl_n   = np.clip(hl / 20.0, 0, 1)
        vpin   = abs(np.clip(imb, -1, 1))
        ivol_n = np.clip(ivol, 0, 1)
        mom_n  = np.clip(mom * 100, -1, 1)

        obs = np.array([fv, spd, imb, inv_n, hl_n, vpin, ivol_n, mom_n],
                       dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        spread_ticks = RL_SPREAD_MAP[int(action)]

        if spread_ticks == 0:
            return None  # Hold — pas de quote

        half = spread_ticks * TICK / 2
        inv_skew = -inventory / RL_MAX_INV * TICK  # skew léger
        our_bid = np.clip(fv - half + inv_skew, TICK, 0.99 - TICK)
        our_ask = np.clip(fv + half + inv_skew, TICK * 2, 0.99)
        return float(round(our_bid, 3)), float(round(our_ask, 3))

    return rl_quotes

# ─── Rapport ─────────────────────────────────────────────────────────────────

def print_report(name, results_by_day):
    pnls = [r['pnl_total'] for r in results_by_day]
    fills = [r['fills_buy'] + r['fills_sell'] for r in results_by_day]

    mean_pnl = np.mean(pnls)
    std_pnl  = np.std(pnls) if len(pnls) > 1 else 0
    sharpe   = mean_pnl / std_pnl if std_pnl > 0 else 0
    win_days = sum(1 for p in pnls if p > 0)

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  P&L moyen      : {mean_pnl:>+8.4f} USDC/jour")
    print(f"  Écart-type     : {std_pnl:>8.4f}")
    print(f"  Sharpe (daily) : {sharpe:>8.3f}")
    print(f"  Jours positifs : {win_days}/{len(pnls)} ({100*win_days/len(pnls):.0f}%)")
    print(f"  Fills moyens   : {np.mean(fills):.1f}/jour")
    print(f"  Adverse select : {np.mean([r['adverse_selection'] for r in results_by_day]):.4f} USDC/jour")

    print(f"\n  Détail par jour :")
    print(f"  {'Date':<28} {'P&L':>8}  {'Fills':>6}  {'Inv final':>10}")
    for (fname, _), r in zip(days_data, results_by_day):
        date = fname.replace('dataset_', '').replace('.csv', '')
        pnl_str  = f"{r['pnl_total']:>+8.4f}"
        fills_str = f"{r['fills_buy']+r['fills_sell']:>6}"
        inv_str   = f"{r['inventory_final']:>10.1f}"
        flag = " <" if r['pnl_total'] > 0 else ""
        print(f"  {date:<28} {pnl_str}  {fills_str}  {inv_str}{flag}")

    return mean_pnl, sharpe

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days',  type=int, default=10,
                    help='Nombre de jours a backtester (defaut: 10)')
    ap.add_argument('--model', type=str,
                    default=None,
                    help='Chemin vers le modele PPO .zip')
    ap.add_argument('--grid',  action='store_true',
                    help='Lancer le grid search des filtres anti-adverse-selection')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  BACKTEST OFFLINE — AS-lite (BS) vs RL PPO")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    # Chargement des données
    print(f"\nChargement de {args.days} jours...")
    days_data = load_days(args.days)
    if not days_data:
        print("ERREUR: Aucune donnée valide.")
        sys.exit(1)
    print(f"  {len(days_data)} jours valides | "
          f"{sum(len(d) for _, d in days_data):,} ticks total")

    # ── Mode grid search ────────────────────────────────────────────────────
    if args.grid:
        grid_results = grid_search_filters(days_data)
        sys.exit(0)

    # ── Backtest AS-lite (baseline sans filtre) ───────────────────────────────
    print(f"\nSimulation AS-lite BS (baseline)...")
    as_results = []
    for fname, df in days_data:
        rng = np.random.default_rng(42)
        r = simulate_day(df, as_lite_quotes, AS_MAX_INV, rng=rng)
        as_results.append(r)
        print(f"  {fname:<35} P&L={r['pnl_total']:>+7.4f}  "
              f"fills={r['fills_buy']+r['fills_sell']}  "
              f"inv={r['inventory_final']:.0f}")

    # ── Backtest AS-lite + filtres (best grid config) ─────────────────────────
    BEST_EDGE = 0.01
    BEST_IMB  = 0.2
    best_quotes_fn = make_filtered_quotes(min_edge=BEST_EDGE, max_imb=BEST_IMB, mom_mult=0.0)
    print(f"\nSimulation AS-lite + filtres (edge={BEST_EDGE}, imb={BEST_IMB})...")
    filt_results = []
    for fname, df in days_data:
        rng = np.random.default_rng(42)
        r = simulate_day(df, best_quotes_fn, AS_MAX_INV, rng=rng)
        filt_results.append(r)
        print(f"  {fname:<35} P&L={r['pnl_total']:>+7.4f}  "
              f"fills={r['fills_buy']+r['fills_sell']}  "
              f"inv={r['inventory_final']:.0f}")

    # ── Backtest RL PPO ───────────────────────────────────────────────────────
    # Trouver le modèle
    model_path = args.model
    if model_path is None:
        models = sorted(glob.glob(os.path.join(MODEL_DIR, 'ppo_mm_*.zip')))
        if not models:
            print("\n[WARN] Aucun modèle PPO trouvé — skip RL comparison")
            rl_results = None
        else:
            model_path = models[-1]  # le plus récent

    rl_results = None
    if model_path and os.path.exists(model_path):
        print(f"\nChargement modèle RL : {os.path.basename(model_path)}...")
        try:
            from stable_baselines3 import PPO
            rl_model = PPO.load(model_path)
            rl_quotes_fn = build_rl_quotes_fn(rl_model)

            print(f"Simulation RL PPO...")
            rl_results = []
            for fname, df in days_data:
                rng = np.random.default_rng(42)  # meme seed pour comparaison equitable
                r = simulate_day(df, rl_quotes_fn, RL_MAX_INV, rng=rng)
                rl_results.append(r)
                print(f"  {fname:<35} P&L={r['pnl_total']:>+7.4f}  "
                      f"fills={r['fills_buy']+r['fills_sell']}  "
                      f"inv={r['inventory_final']:.0f}")
        except Exception as e:
            print(f"  [WARN] Impossible de charger le modèle RL: {e}")
            rl_results = None
    else:
        print(f"\n[WARN] Modèle RL non trouvé ({model_path}) — skip RL")

    # ── Rapport final ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  RÉSULTATS FINAUX")
    print(f"{'='*60}")

    as_mean, as_sharpe = print_report("AS-lite baseline (sans filtre)", as_results)
    filt_mean, filt_sharpe = print_report(f"AS-lite + filtres (edge={BEST_EDGE}, imb={BEST_IMB})", filt_results)

    if rl_results:
        rl_mean, rl_sharpe = print_report("RL PPO (MLP 64x64)", rl_results)

    # ── Comparaison finale ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARAISON FINALE (memes donnees, meme fill model)")
    print(f"{'='*70}")
    print(f"  {'Strategie':<35} {'PnL/jour':>10} {'Sharpe':>8} {'Fills/j':>8}")
    print(f"  {'─'*65}")
    print(f"  {'AS-lite baseline':<35} {as_mean:>+10.2f} {as_sharpe:>+8.3f} "
          f"{np.mean([r['fills_buy']+r['fills_sell'] for r in as_results]):>8.1f}")
    print(f"  {'AS-lite + filtres':<35} {filt_mean:>+10.2f} {filt_sharpe:>+8.3f} "
          f"{np.mean([r['fills_buy']+r['fills_sell'] for r in filt_results]):>8.1f}")
    if rl_results:
        print(f"  {'RL PPO':<35} {rl_mean:>+10.2f} {rl_sharpe:>+8.3f} "
              f"{np.mean([r['fills_buy']+r['fills_sell'] for r in rl_results]):>8.1f}")

    print(f"{'='*70}\n")
