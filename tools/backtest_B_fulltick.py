import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, numpy as np, pandas as pd
from itertools import product

DATA_DIR  = "data"
QUOTE_SIZE = 5.0
TICK       = 0.01
OOS_DAYS   = 7

USE_COLS = {'best_bid','best_ask','fv_theoretical','bid_vol_1','ask_vol_1',
            'imbalance','hours_left','implied_vol','momentum'}

def load_days():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f]
    days = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False,
                             usecols=lambda c: c in USE_COLS)
            for col in df.select_dtypes(include='object').columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['best_bid','best_ask','fv_theoretical'])
            df = df[df['best_bid'] > 0]
            df = df[df['fv_theoretical'].between(0.05, 0.95)]
            if 'hours_left' in df.columns:
                df = df[df['hours_left'] > 0.5]
            if len(df) >= 50:
                days.append((os.path.basename(f), df))
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")
    return days

def simulate_day_b(df, spread_base=0.10, vol_threshold=0.45, vol_mult=2.0, max_imb=0.20, min_edge=0.005, max_inv=50.0):
    n = len(df)
    inv, cost, realized, fills = 0.0, 0.0, 0.0, 0

    # Convertir en numpy pour aller vite
    bids     = df['best_bid'].values.astype(float)
    asks     = df['best_ask'].values.astype(float)
    fvs      = df['fv_theoretical'].values.astype(float)
    imbs     = df['imbalance'].values.astype(float) if 'imbalance' in df.columns else np.zeros(n)
    ivols    = df['implied_vol'].values.astype(float) if 'implied_vol' in df.columns else np.full(n, 0.40)

    for t in range(n - 1):
        bid, ask = bids[t], asks[t]
        fv = fvs[t]
        imb = imbs[t]
        ivol = ivols[t]

        nxt_bid, nxt_ask = bids[t+1], asks[t+1]

        if bid <= 0 or ask <= 0 or nxt_bid <= 0 or nxt_ask <= 0:
            continue
        if abs(imb) > max_imb:
            continue

        market_mid = (bid + ask) / 2
        fv_anchored = 0.5 * fv + 0.5 * market_mid

        if abs(fv_anchored - market_mid) < min_edge:
            continue

        # Regime B : Deribit IV > seuil -> spread x mult
        hs = spread_base
        if ivol > vol_threshold:
            hs = spread_base * vol_mult

        inv_ratio = max(-1.0, min(1.0, inv / 50.0))
        center = fv_anchored - inv_ratio * 0.02

        our_bid = round(max(TICK, center - hs), 2)
        our_ask = round(min(1.0 - TICK, center + hs), 2)

        if our_bid >= nxt_ask and inv < max_inv:
            sh = min(QUOTE_SIZE / our_bid, max_inv - inv)
            if sh >= 1:
                inv += sh; cost += our_bid * sh; fills += 1
        elif our_ask <= nxt_bid and inv > 0:
            sh = min(QUOTE_SIZE / our_ask, inv)
            if sh >= 1:
                avg = cost / inv
                realized += sh * (our_ask - avg)
                cost -= avg * sh; inv = max(0, inv - sh)
                if inv < 0.01: cost = 0
                fills += 1

    return realized, fills


print("\n" + "="*65)
print("  BACKTEST PISTE B FULL TICKS")
print("="*65)

all_days = load_days()
days_is  = all_days[:-OOS_DAYS]
days_oos = all_days[-OOS_DAYS:]
print(f"  {len(days_is)} jours IS | {len(days_oos)} jours OOS")
print(f"  IS: {days_is[0][0]} -> {days_is[-1][0]}")
print(f"  OOS: {days_oos[0][0]} -> {days_oos[-1][0]}")
print(f"  Ticks total: {sum(len(d) for _, d in all_days):,}")

# Grid sur vol_threshold et vol_mult
vol_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
vol_mults      = [1.5, 2.0, 2.5, 3.0]

print(f"\n{'Config':>25} {'IS P&L':>10} {'OOS P&L':>10} {'OOS fills':>10}")
print("-" * 60)

# Baseline d'abord
bl_is  = [simulate_day_b(df, vol_threshold=999)[0] for _, df in days_is]
bl_oos = [simulate_day_b(df, vol_threshold=999)[0] for _, df in days_oos]
bl_oos_f = [simulate_day_b(df, vol_threshold=999)[1] for _, df in days_oos]
print(f"  {'BASELINE (no regime)':>23}  {np.mean(bl_is):>+8.3f}  {np.mean(bl_oos):>+8.3f}  {np.mean(bl_oos_f):>8.1f}/j")

best_oos = -999
best_cfg = None
results = []
for vt, vm in product(vol_thresholds, vol_mults):
    is_pnls  = [simulate_day_b(df, vol_threshold=vt, vol_mult=vm)[0] for _, df in days_is]
    oos_res  = [simulate_day_b(df, vol_threshold=vt, vol_mult=vm) for _, df in days_oos]
    oos_pnls = [r[0] for r in oos_res]
    oos_fills = [r[1] for r in oos_res]
    m_oos = np.mean(oos_pnls)
    print(f"  vol>{vt:.2f}  x{vm:.1f}         {np.mean(is_pnls):>+8.3f}  {m_oos:>+8.3f}  {np.mean(oos_fills):>8.1f}/j")
    results.append((vt, vm, np.mean(is_pnls), m_oos, np.mean(oos_fills)))
    if m_oos > best_oos:
        best_oos = m_oos
        best_cfg = (vt, vm)

print(f"\n>>> BEST OOS : vol>{best_cfg[0]:.2f} x{best_cfg[1]:.1f} -> {best_oos:+.3f}$/j")

# Detail jour par jour pour le best config
print(f"\n--- Detail OOS jour par jour (vol>{best_cfg[0]:.2f} x{best_cfg[1]:.1f}) ---")
for fname, df in days_oos:
    pnl, fills = simulate_day_b(df, vol_threshold=best_cfg[0], vol_mult=best_cfg[1])
    print(f"  {fname:<35} P&L={pnl:>+8.3f}  fills={fills}")

print(f"\n{'='*65}")
