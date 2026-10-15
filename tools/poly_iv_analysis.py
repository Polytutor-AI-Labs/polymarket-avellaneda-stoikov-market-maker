#!/usr/bin/env python3
"""
poly_iv_analysis.py — Back-solve la volatilite implicite Polymarket
et compare avec l'IV Deribit (colonne implied_vol).

Question : est-ce que Polymarket overvalue la vol ? A quelle heure/conditions ?
"""

import sys, os, math, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from strategy.pricing import calculate_black_scholes_binary

DATA_DIR = "data"


def back_solve_implied_vol(market_mid, btc_price, strike, T_years,
                           lo=0.01, hi=5.0, tol=1e-4, max_iter=50):
    """Bisection pour trouver sigma tel que BS(S, K, T, sigma) = market_mid."""
    if T_years <= 1e-6 or market_mid <= 0.01 or market_mid >= 0.99:
        return float('nan')
    if btc_price <= 0 or strike <= 0:
        return float('nan')

    for _ in range(max_iter):
        mid_vol = (lo + hi) / 2.0
        bs_price = calculate_black_scholes_binary(btc_price, strike, T_years, mid_vol)
        if abs(bs_price - market_mid) < tol:
            return mid_vol
        if bs_price > market_mid:
            # BS gives higher price -> vol too high (for S > K) or too low
            # For binary: higher vol -> price closer to 0.5
            # Need to check direction
            if market_mid > 0.5:
                hi = mid_vol  # price too low relative to market, reduce vol
            else:
                lo = mid_vol
        else:
            if market_mid > 0.5:
                lo = mid_vol
            else:
                hi = mid_vol
    return (lo + hi) / 2.0


def analyze():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f]

    USE_COLS = {'btc_price', 'best_bid', 'best_ask', 'mid_price',
                'implied_vol', 'hours_left', 'market_strike', 'timestamp'}

    all_rows = []

    print(f"Chargement de {len(files)} fichiers...")
    for f in files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False,
                             usecols=lambda c: c in USE_COLS)
            for col in ['btc_price', 'best_bid', 'best_ask', 'mid_price',
                         'implied_vol', 'hours_left', 'market_strike']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            required = ['btc_price', 'best_bid', 'best_ask', 'implied_vol',
                         'hours_left', 'market_strike']
            if not all(c in df.columns for c in required):
                print(f"  [SKIP] {fname} — colonnes manquantes")
                continue

            df = df.dropna(subset=required)
            df = df[df['hours_left'] > 0.5]
            df = df[df['best_bid'] > 0.05]
            df = df[df['best_ask'] < 0.95]
            df = df[df['market_strike'] > 0]

            # Sous-echantillonner a 500 lignes/jour pour la vitesse du solver
            step = max(1, len(df) // 500)
            df = df.iloc[::step].reset_index(drop=True)

            if len(df) < 10:
                print(f"  [SKIP] {fname} — trop peu de lignes ({len(df)})")
                continue

            # Calculer market_mid et poly_iv
            df['market_mid'] = (df['best_bid'] + df['best_ask']) / 2.0
            df['T_years'] = df['hours_left'] / 8760.0

            poly_ivs = []
            for _, row in df.iterrows():
                piv = back_solve_implied_vol(
                    row['market_mid'], row['btc_price'],
                    row['market_strike'], row['T_years'])
                poly_ivs.append(piv)
            df['poly_iv'] = poly_ivs

            df['date'] = fname.replace('dataset_', '').replace('.csv', '')

            # Extraire l'heure si timestamp disponible
            if 'timestamp' in df.columns:
                ts = pd.to_datetime(df['timestamp'], errors='coerce', format='ISO8601')
                df['hour'] = ts.dt.hour
            else:
                df['hour'] = float('nan')

            valid = df.dropna(subset=['poly_iv'])
            valid = valid[valid['poly_iv'].between(0.05, 4.0)]
            all_rows.append(valid)
            print(f"  {fname}: {len(valid)} points, "
                  f"poly_iv={valid['poly_iv'].mean():.3f}, "
                  f"deribit_iv={valid['implied_vol'].mean():.3f}")

        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    if not all_rows:
        print("ERREUR: aucune donnee valide.")
        return

    data = pd.concat(all_rows, ignore_index=True)
    data['iv_ratio'] = data['poly_iv'] / data['implied_vol']
    data['iv_diff'] = data['poly_iv'] - data['implied_vol']

    # === RESULTATS ===
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n{'='*70}")
    out(f"  ANALYSE POLY_IV — Polymarket vs Deribit implied vol")
    out(f"  {len(data)} points sur {data['date'].nunique()} jours")
    out(f"{'='*70}")

    out(f"\n--- Stats globales ---")
    out(f"  Poly IV  moyenne : {data['poly_iv'].mean():.4f} (median: {data['poly_iv'].median():.4f})")
    out(f"  Deribit IV moyenne : {data['implied_vol'].mean():.4f} (median: {data['implied_vol'].median():.4f})")
    out(f"  Ratio poly/deribit : {data['iv_ratio'].mean():.3f} (median: {data['iv_ratio'].median():.3f})")
    out(f"  Diff poly - deribit: {data['iv_diff'].mean():+.4f} (median: {data['iv_diff'].median():+.4f})")

    pct_over = (data['poly_iv'] > data['implied_vol']).mean() * 100
    out(f"  % ticks ou poly_iv > deribit_iv : {pct_over:.1f}%")

    out(f"\n--- Par jour ---")
    out(f"  {'Date':<14} {'Poly IV':>8} {'Deribit IV':>10} {'Ratio':>7} {'Diff':>8} {'N':>5}")
    out(f"  {'─'*55}")
    by_day = data.groupby('date').agg(
        poly_iv=('poly_iv', 'mean'),
        deribit_iv=('implied_vol', 'mean'),
        ratio=('iv_ratio', 'mean'),
        diff=('iv_diff', 'mean'),
        n=('poly_iv', 'count'),
    )
    for date, row in by_day.iterrows():
        flag = " <<" if row['ratio'] > 1.5 else ""
        out(f"  {date:<14} {row['poly_iv']:>8.4f} {row['deribit_iv']:>10.4f} "
            f"{row['ratio']:>7.2f} {row['diff']:>+8.4f} {row['n']:>5.0f}{flag}")

    # Par heure
    if data['hour'].notna().any():
        out(f"\n--- Par heure (UTC) ---")
        out(f"  {'Heure':>5} {'Poly IV':>8} {'Deribit IV':>10} {'Ratio':>7} {'N':>5}")
        out(f"  {'─'*40}")
        by_hour = data.dropna(subset=['hour']).groupby('hour').agg(
            poly_iv=('poly_iv', 'mean'),
            deribit_iv=('implied_vol', 'mean'),
            ratio=('iv_ratio', 'mean'),
            n=('poly_iv', 'count'),
        )
        for hour, row in by_hour.iterrows():
            flag = " <<" if row['ratio'] > 1.5 else ""
            out(f"  {int(hour):>5} {row['poly_iv']:>8.4f} {row['deribit_iv']:>10.4f} "
                f"{row['ratio']:>7.2f} {row['n']:>5.0f}{flag}")

    # Par tranche de hours_left
    out(f"\n--- Par temps restant ---")
    out(f"  {'Hours left':>10} {'Poly IV':>8} {'Deribit IV':>10} {'Ratio':>7} {'N':>5}")
    out(f"  {'─'*45}")
    data['hl_bucket'] = pd.cut(data['hours_left'],
                                bins=[0, 2, 4, 8, 12, 24, 100],
                                labels=['0-2h', '2-4h', '4-8h', '8-12h', '12-24h', '24h+'])
    by_hl = data.groupby('hl_bucket', observed=True).agg(
        poly_iv=('poly_iv', 'mean'),
        deribit_iv=('implied_vol', 'mean'),
        ratio=('iv_ratio', 'mean'),
        n=('poly_iv', 'count'),
    )
    for bucket, row in by_hl.iterrows():
        flag = " <<" if row['ratio'] > 1.5 else ""
        out(f"  {str(bucket):>10} {row['poly_iv']:>8.4f} {row['deribit_iv']:>10.4f} "
            f"{row['ratio']:>7.2f} {row['n']:>5.0f}{flag}")

    # Conclusion
    out(f"\n{'='*70}")
    out(f"  CONCLUSION")
    out(f"{'='*70}")
    if pct_over > 60:
        out(f"  Polymarket OVERVALUE la vol {pct_over:.0f}% du temps (ratio moyen {data['iv_ratio'].mean():.2f}x)")
        out(f"  -> Opportunite : vendre de la vol (shorter les options OTM)")
    elif pct_over < 40:
        out(f"  Polymarket UNDERVALUE la vol {100-pct_over:.0f}% du temps")
        out(f"  -> Opportunite : acheter de la vol (acheter les options OTM)")
    else:
        out(f"  Poly_iv et Deribit_iv sont proches (ratio {data['iv_ratio'].mean():.2f}x)")
        out(f"  -> Pas d'edge systematique sur la vol")
    out(f"{'='*70}\n")

    # Sauvegarder
    with open('results/poly_iv_analysis.txt', 'w') as fp:
        fp.write('\n'.join(lines))
    print("Resultats sauvegardes dans results/poly_iv_analysis.txt")


if __name__ == '__main__':
    analyze()
