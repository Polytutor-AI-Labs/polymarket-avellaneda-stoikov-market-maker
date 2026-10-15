#!/usr/bin/env python3
"""
backtest_AB.py — Grid search stratégies A (spread dynamique/T) et B (regime detection).

Piste A : Spread = f(hours_left) — ne pas garder un spread fixe toute la journée.
           Les options binaires ont une explosion du gamma risk près de l'expiry.
           Tester : spread_early (T>12h) / spread_mid (4-12h) / spread_late (<4h)

Piste B : Regime detection — ne pas quoter quand BTC trend fort.
           Filtres testés : momentum + implied_vol + combo.

Split IS/OOS chronologique :
  - IS  : 27 premiers jours (entraînement)
  - OOS :  7 derniers jours (validation)

Usage : python3 tools/backtest_AB.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

DATA_DIR  = "data"
QUOTE_SIZE = 5.0
TICK       = 0.01
OOS_DAYS   = 7    # derniers jours = validation
TOP_N      = 15   # afficher top N configs

USE_COLS = {
    'best_bid', 'best_ask', 'fv_theoretical', 'bid_vol_1', 'ask_vol_1',
    'imbalance', 'hours_left', 'implied_vol', 'momentum',
}

# ─── Chargement ───────────────────────────────────────────────────────────────

def load_days(max_ticks_per_day=2000):
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f]
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
            step = max(1, len(df) // max_ticks_per_day)
            df = df.iloc[::step].reset_index(drop=True)
            if len(df) >= 50:
                days.append((os.path.basename(f), df))
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")
    return days

# ─── Simulateur de fills ───────────────────────────────────────────────────────

def simulate_day(df, get_quotes_fn, max_inv=50.0):
    n = len(df)
    inventory  = 0.0
    cost_basis = 0.0
    realized   = 0.0
    fills_buy  = 0
    fills_sell = 0

    # Pre-extract numpy arrays for speed
    fvs  = df['fv_theoretical'].values.astype(np.float64)
    bids = df['best_bid'].values.astype(np.float64)
    asks = df['best_ask'].values.astype(np.float64)
    imbs = df['imbalance'].values.astype(np.float64) if 'imbalance' in df.columns else np.zeros(n)
    hls  = df['hours_left'].values.astype(np.float64) if 'hours_left' in df.columns else np.full(n, 5.0)
    ivols = df['implied_vol'].values.astype(np.float64) if 'implied_vol' in df.columns else np.full(n, 0.5)
    moms = df['momentum'].values.astype(np.float64) if 'momentum' in df.columns else np.zeros(n)

    for t in range(n - 1):
        bid, ask, fv = bids[t], asks[t], fvs[t]
        nxt_bid, nxt_ask = bids[t+1], asks[t+1]

        if bid <= 0 or ask <= 0 or nxt_bid <= 0 or nxt_ask <= 0:
            continue

        market_mid = (bid + ask) / 2
        fv_anchored = 0.5 * fv + 0.5 * market_mid

        result = get_quotes_fn(fv_anchored, bid, ask, imbs[t], hls[t], ivols[t], moms[t], inventory, t)
        if result is None:
            continue
        our_bid, our_ask = result

        if our_bid >= nxt_ask and inventory < max_inv:
            shares = QUOTE_SIZE / our_bid if our_bid > 0 else 0
            shares = min(shares, max_inv - inventory)
            if shares >= 1.0:
                inventory  += shares
                cost_basis += our_bid * shares
                fills_buy  += 1
        elif our_ask <= nxt_bid and inventory > 0:
            shares = min(QUOTE_SIZE / our_ask if our_ask > 0 else 0, inventory)
            if shares >= 1.0:
                avg = cost_basis / inventory if inventory > 0 else 0
                realized   += shares * (our_ask - avg)
                cost_basis -= avg * shares
                inventory   = max(0.0, inventory - shares)
                if inventory <= 0.01:
                    cost_basis = 0.0
                fills_sell += 1

    return {
        'pnl': realized,
        'fills': fills_buy + fills_sell,
        'fills_buy': fills_buy,
        'fills_sell': fills_sell,
        'inv_final': inventory,
    }

# ─── PISTE A : Spread dynamique f(T) ─────────────────────────────────────────

def make_spread_T_fn(spread_early, spread_mid, spread_late, spread_stop,
                     min_edge=0.005, max_imb=0.2,
                     t_early=12.0, t_mid=4.0, t_late=1.5):
    """
    Spread par paliers selon hours_left.
    spread_stop = stop quoting (None ou float très large).
    """
    def quote_fn(fv, bid, ask, imb, hl, ivol, mom, inv, t):
        # Filtre imbalance
        if abs(imb) > max_imb:
            return None
        # Filtre edge minimum
        market_mid = (bid + ask) / 2
        if abs(fv - market_mid) < min_edge:
            return None
        # Spread selon T
        if hl > t_early:
            hs = spread_early
        elif hl > t_mid:
            hs = spread_mid
        elif hl > t_late:
            hs = spread_late
        else:
            if spread_stop is None:
                return None   # stop quoting très près de l'expiry
            hs = spread_stop

        inv_ratio = max(-1.0, min(1.0, inv / 50.0))
        center = fv - inv_ratio * 0.02

        our_bid = round(max(TICK, center - hs), 2)
        our_ask = round(min(1.0 - TICK, center + hs), 2)
        return our_bid, our_ask

    return quote_fn

# ─── PISTE B : Regime detection ───────────────────────────────────────────────

def make_regime_fn(mom_threshold, vol_threshold, regime_action,
                   spread_base=0.10, min_edge=0.005, max_imb=0.2):
    """
    Si momentum fort OU vol implicite haute → régime tendanciel.
    regime_action : "skip" (ne pas quoter) ou "double" (spread × 2)
    """
    def quote_fn(fv, bid, ask, imb, hl, ivol, mom, inv, t):
        if abs(imb) > max_imb:
            return None
        market_mid = (bid + ask) / 2
        if abs(fv - market_mid) < min_edge:
            return None

        # Détection régime
        trending = (abs(mom) > mom_threshold) or (ivol > vol_threshold)

        if trending:
            if regime_action == "skip":
                return None
            hs = spread_base * 2.0   # double spread
        else:
            hs = spread_base

        inv_ratio = max(-1.0, min(1.0, inv / 50.0))
        center = fv - inv_ratio * 0.02
        our_bid = round(max(TICK, center - hs), 2)
        our_ask = round(min(1.0 - TICK, center + hs), 2)
        return our_bid, our_ask

    return quote_fn

# ─── PISTE A+B combinée ───────────────────────────────────────────────────────

def make_combined_fn(spread_early, spread_mid, spread_late,
                     mom_threshold, vol_threshold,
                     min_edge=0.005, max_imb=0.2):
    """Meilleure config A + meilleure config B combinées."""
    def quote_fn(fv, bid, ask, imb, hl, ivol, mom, inv, t):
        if abs(imb) > max_imb:
            return None
        market_mid = (bid + ask) / 2
        if abs(fv - market_mid) < min_edge:
            return None

        # Regime check
        trending = (abs(mom) > mom_threshold) or (ivol > vol_threshold)
        if trending and hl < 4.0:   # trend + peu de temps = très dangereux
            return None

        # Spread selon T
        if hl > 12.0:
            hs = spread_early
        elif hl > 4.0:
            hs = spread_mid * (2.0 if trending else 1.0)
        else:
            hs = spread_late * (1.5 if trending else 1.0)

        inv_ratio = max(-1.0, min(1.0, inv / 50.0))
        center = fv - inv_ratio * 0.02
        our_bid = round(max(TICK, center - hs), 2)
        our_ask = round(min(1.0 - TICK, center + hs), 2)
        return our_bid, our_ask

    return quote_fn

# ─── Runner ───────────────────────────────────────────────────────────────────

def run_grid(days_is, days_oos, configs, label=""):
    results = []
    n = len(configs)
    for i, (name, fn) in enumerate(configs):
        if i % max(1, n // 10) == 0:
            print(f"  [{i+1}/{n}] {name[:60]}")

        is_res  = [simulate_day(df, fn) for _, df in days_is]
        oos_res = [simulate_day(df, fn) for _, df in days_oos]

        is_pnls   = [r['pnl'] for r in is_res]
        oos_pnls  = [r['pnl'] for r in oos_res]

        results.append({
            'config': name,
            'is_pnl':    np.mean(is_pnls),
            'oos_pnl':   np.mean(oos_pnls),
            'is_fills':  np.mean([r['fills'] for r in is_res]),
            'oos_fills': np.mean([r['fills'] for r in oos_res]),
            'is_sharpe':  np.mean(is_pnls) / (np.std(is_pnls) + 1e-9),
            'oos_sharpe': np.mean(oos_pnls) / (np.std(oos_pnls) + 1e-9),
        })

    df = pd.DataFrame(results).sort_values('oos_pnl', ascending=False)
    return df

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\n{'='*65}")
    print(f"  BACKTEST A+B — Spread dynamique T + Regime Detection")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*65}")

    print("\nChargement des données...")
    all_days = load_days(max_ticks_per_day=10000)  # 10k ticks/jour (5x plus que AWS, bon compromis)
    if not all_days:
        print("ERREUR: aucune donnée.")
        sys.exit(1)

    days_is  = all_days[:-OOS_DAYS]
    days_oos = all_days[-OOS_DAYS:]
    print(f"  {len(days_is)} jours IS | {len(days_oos)} jours OOS")
    print(f"  IS  : {days_is[0][0][:18]} → {days_is[-1][0][:18]}")
    print(f"  OOS : {days_oos[0][0][:18]} → {days_oos[-1][0][:18]}")

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n[Baseline] AS-lite fixe hs=10c + filtres validés...")
    baseline_fn = make_spread_T_fn(0.10, 0.10, 0.10, 0.10)
    base_is  = [simulate_day(df, baseline_fn)['pnl'] for _, df in days_is]
    base_oos = [simulate_day(df, baseline_fn)['pnl'] for _, df in days_oos]
    print(f"  Baseline IS  : {np.mean(base_is):>+7.3f} $/j  Sharpe={np.mean(base_is)/(np.std(base_is)+1e-9):.3f}")
    print(f"  Baseline OOS : {np.mean(base_oos):>+7.3f} $/j  Sharpe={np.mean(base_oos)/(np.std(base_oos)+1e-9):.3f}")

    # ── Piste A : Spread dynamique T ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("PISTE A — Grid search spread f(T)")
    print(f"{'─'*65}")

    spreads_early = [0.06, 0.08, 0.10, 0.12]   # T > 12h
    spreads_mid   = [0.08, 0.10, 0.12, 0.15]   # T 4-12h
    spreads_late  = [0.10, 0.12, 0.15, 0.20, None]  # T < 4h (None = stop)

    configs_A = []
    for se, sm, sl in product(spreads_early, spreads_mid, spreads_late):
        sl_val = sl if sl is not None else 99.0
        name = f"A: early={int((se or 0)*100)}c mid={int(sm*100)}c late={'STOP' if sl is None else str(int(sl_val*100))+'c'}"
        fn = make_spread_T_fn(se, sm, sl_val if sl is not None else 99, None if sl is None else sl_val)
        configs_A.append((name, fn))

    print(f"  {len(configs_A)} configurations...")
    res_A = run_grid(days_is, days_oos, configs_A, "A")

    print(f"\n  TOP {TOP_N} configs piste A (triées par OOS P&L) :")
    print(f"  {'Config':<45} {'IS':>8} {'OOS':>8} {'Fills OOS':>10}")
    print(f"  {'-'*75}")
    for _, row in res_A.head(TOP_N).iterrows():
        print(f"  {row['config']:<45} {row['is_pnl']:>+7.3f} {row['oos_pnl']:>+7.3f}  {row['oos_fills']:>6.1f}/j")

    best_A = res_A.iloc[0]
    print(f"\n  >>> BEST A : {best_A['config']}")
    print(f"      OOS PnL = {best_A['oos_pnl']:+.3f}$/j | Sharpe = {best_A['oos_sharpe']:.3f}")

    # ── Piste B : Regime detection ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("PISTE B — Grid search regime detection")
    print(f"{'─'*65}")

    mom_thresholds  = [0.001, 0.002, 0.003, 0.005, 0.010]
    vol_thresholds  = [0.45, 0.55, 0.65, 0.80, 999.0]   # 999 = désactivé
    regime_actions  = ["skip", "double"]

    configs_B = []
    for mt, vt, ra in product(mom_thresholds, vol_thresholds, regime_actions):
        vt_str = "OFF" if vt > 10 else f"{vt:.2f}"
        name = f"B: mom>{mt:.3f} vol>{vt_str} action={ra}"
        fn = make_regime_fn(mt, vt, ra)
        configs_B.append((name, fn))

    print(f"  {len(configs_B)} configurations...")
    res_B = run_grid(days_is, days_oos, configs_B, "B")

    print(f"\n  TOP {TOP_N} configs piste B (triées par OOS P&L) :")
    print(f"  {'Config':<50} {'IS':>8} {'OOS':>8} {'Fills OOS':>10}")
    print(f"  {'-'*80}")
    for _, row in res_B.head(TOP_N).iterrows():
        print(f"  {row['config']:<50} {row['is_pnl']:>+7.3f} {row['oos_pnl']:>+7.3f}  {row['oos_fills']:>6.1f}/j")

    best_B = res_B.iloc[0]
    print(f"\n  >>> BEST B : {best_B['config']}")
    print(f"      OOS PnL = {best_B['oos_pnl']:+.3f}$/j | Sharpe = {best_B['oos_sharpe']:.3f}")

    # ── Combo A+B ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("COMBO A+B — Meilleure config A × meilleure config B")
    print(f"{'─'*65}")

    # Extraire les params best A (approximation depuis le nom)
    best_A_name = best_A['config']
    # Tester quelques combos autour des meilleurs params
    combo_spreads_early = [0.06, 0.08, 0.10]
    combo_spreads_mid   = [0.08, 0.10, 0.12]
    combo_spreads_late  = [0.12, 0.15, 0.20]
    combo_mom    = [0.002, 0.003, 0.005]
    combo_vol    = [0.55, 0.65, 999.0]

    configs_C = []
    for se, sm, sl, mt, vt in product(combo_spreads_early, combo_spreads_mid,
                                       combo_spreads_late, combo_mom, combo_vol):
        vt_str = "OFF" if vt > 10 else f"{vt:.2f}"
        name = f"AB: e={int(se*100)}c m={int(sm*100)}c l={int(sl*100)}c mom>{mt:.3f} vol>{vt_str}"
        fn = make_combined_fn(se, sm, sl, mt, vt)
        configs_C.append((name, fn))

    print(f"  {len(configs_C)} configurations...")
    res_C = run_grid(days_is, days_oos, configs_C, "AB")

    print(f"\n  TOP {TOP_N} configs COMBO A+B :")
    print(f"  {'Config':<60} {'IS':>8} {'OOS':>8} {'Fills OOS':>10}")
    print(f"  {'-'*90}")
    for _, row in res_C.head(TOP_N).iterrows():
        print(f"  {row['config']:<60} {row['is_pnl']:>+7.3f} {row['oos_pnl']:>+7.3f}  {row['oos_fills']:>6.1f}/j")

    best_C = res_C.iloc[0]

    # ── Résumé final ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*65}")
    print(f"  {'Stratégie':<35} {'IS P&L':>10} {'OOS P&L':>10} {'OOS Sharpe':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Baseline (hs=10c fixe)':<35} {np.mean(base_is):>+9.3f} {np.mean(base_oos):>+9.3f}  {np.mean(base_oos)/(np.std(base_oos)+1e-9):>10.3f}")
    print(f"  {'Best A (spread dyn T)':<35} {best_A['is_pnl']:>+9.3f} {best_A['oos_pnl']:>+9.3f}  {best_A['oos_sharpe']:>10.3f}")
    print(f"  {'Best B (regime detect)':<35} {best_B['is_pnl']:>+9.3f} {best_B['oos_pnl']:>+9.3f}  {best_B['oos_sharpe']:>10.3f}")
    print(f"  {'Best A+B (combo)':<35} {best_C['is_pnl']:>+9.3f} {best_C['oos_pnl']:>+9.3f}  {best_C['oos_sharpe']:>10.3f}")
    print(f"\n  Best A config : {best_A['config']}")
    print(f"  Best B config : {best_B['config']}")
    print(f"  Best combo    : {best_C['config']}")
    print(f"\n{'='*65}\n")

    # Sauvegarde
    out = "data/backtest_AB_results.csv"
    pd.concat([res_A, res_B, res_C]).to_csv(out, index=False)
    print(f"Résultats complets sauvegardés dans {out}")
