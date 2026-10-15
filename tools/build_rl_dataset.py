#!/usr/bin/env python3
"""
build_rl_dataset.py — Consolidation et validation du dataset pour RL.

Pipeline Jane Street style :
  1. Audit de qualité : gaps, schéma, valeurs aberrantes (streaming, 1 fichier à la fois)
  2. Feature engineering : VPIN proxy, vol réalisée, fv_velocity, session_progress
  3. Export streaming : écriture incrémentale → pas d'OOM
  4. Stats descriptives pour calibrer l'espace d'état du RL

Usage:
    python3 tools/build_rl_dataset.py              # audit + stats (derniers 30 jours)
    python3 tools/build_rl_dataset.py --days 60    # N derniers jours
    python3 tools/build_rl_dataset.py --export     # exporte MASTER_DATASET_RL.csv
    python3 tools/build_rl_dataset.py --export --days 30
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

DATA_DIR   = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "MASTER_DATASET_RL.csv")

REQUIRED_COLS = [
    'timestamp', 'btc_price', 'best_bid', 'best_ask',
    'bid_vol_1', 'ask_vol_1', 'imbalance',
    'fv_theoretical', 'hours_left',
]

# Features utilisées pour l'entraînement RL (espace d'état)
RL_STATE_COLS = [
    'btc_price', 'best_bid', 'best_ask', 'bid_vol_1', 'ask_vol_1',
    'imbalance', 'fv_theoretical', 'implied_vol', 'spread', 'hours_left',
    'momentum', 'vpin_proxy', 'vol_realized_10m', 'fv_velocity',
    'funding_rate', 'session_progress',
    'bid_price_2', 'bid_vol_2', 'ask_price_2', 'ask_vol_2',
    'last_trade_price', 'last_trade_size',
]


# ── AUDIT STREAMING (1 fichier à la fois) ─────────────────────────────────────

def audit_file(filepath):
    """Audite un CSV journalier. Charge → audite → libère. Retourne dict."""
    date_str = os.path.basename(filepath).replace('dataset_', '').replace('.csv', '')
    result = {
        'date': date_str, 'path': filepath,
        'n_rows': 0, 'n_cols': 0,
        'has_required': False, 'has_full_schema': False,
        'avg_spread': None, 'issues': [],
    }
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False, nrows=5)
        cols = set(df.columns)
        result['n_cols'] = len(cols)

        missing_req = [c for c in REQUIRED_COLS if c not in cols]
        if missing_req:
            result['issues'].append(f"Manquantes: {missing_req}")
            return result
        result['has_required'] = True
        result['has_full_schema'] = len(cols) >= 35

        # Stats légères (re-charger complet)
        df_full = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False,
                              usecols=lambda c: c in {'fv_theoretical', 'best_bid', 'best_ask'})
        result['n_rows'] = len(df_full)

        if 'fv_theoretical' in df_full.columns:
            fv = pd.to_numeric(df_full['fv_theoretical'], errors='coerce')
            valid_pct = fv.between(0.01, 0.99).mean() * 100
            result['fv_valid_pct'] = round(valid_pct, 1)
            if valid_pct < 50:
                result['issues'].append(f"FV valide seulement {valid_pct:.0f}%")

        if 'best_bid' in df_full.columns and 'best_ask' in df_full.columns:
            bid = pd.to_numeric(df_full['best_bid'], errors='coerce')
            ask = pd.to_numeric(df_full['best_ask'], errors='coerce')
            spread = ask - bid
            result['avg_spread'] = round(spread.mean(), 4)
            neg = (spread < 0).sum()
            if neg > 0:
                result['issues'].append(f"{neg} spreads négatifs")

        if result['n_rows'] < 1000:
            result['issues'].append(f"Peu de lignes: {result['n_rows']}")

    except Exception as e:
        result['issues'].append(f"Erreur lecture: {e}")

    return result


# ── FEATURE ENGINEERING (sur un DataFrame quotidien) ─────────────────────────

def add_features(df):
    """Ajoute les features dérivées pour le RL. Opère in-place sur le df quotidien."""

    # VPIN proxy (flow toxicity)
    if 'last_trade_side' in df.columns and 'last_trade_size' in df.columns:
        sz  = pd.to_numeric(df['last_trade_size'], errors='coerce').fillna(0)
        sell_vol = sz.where(df['last_trade_side'] == 'SELL', 0)
        buy_vol  = sz.where(df['last_trade_side'] == 'BUY',  0)
        w = 50
        s_cum = sell_vol.rolling(w, min_periods=1).sum()
        b_cum = buy_vol.rolling(w, min_periods=1).sum()
        tot   = (s_cum + b_cum).replace(0, np.nan)
        df['vpin_proxy'] = ((s_cum - b_cum).abs() / tot).fillna(0).round(4)
    else:
        df['vpin_proxy'] = 0.0

    # Volatilité réalisée BTC (10 min rolling ≈ 600 ticks à 1Hz)
    if 'btc_price' in df.columns:
        ret = pd.to_numeric(df['btc_price'], errors='coerce').pct_change()
        df['vol_realized_10m'] = (ret.rolling(600, min_periods=10).std() * np.sqrt(86400)).fillna(0).round(6)
    else:
        df['vol_realized_10m'] = 0.0

    # Vitesse de changement de FV
    if 'fv_theoretical' in df.columns:
        fv = pd.to_numeric(df['fv_theoretical'], errors='coerce')
        df['fv_velocity'] = fv.diff().fillna(0).round(6)
    else:
        df['fv_velocity'] = 0.0

    # Progression de session [0,1]
    if 'hours_left' in df.columns:
        hl = pd.to_numeric(df['hours_left'], errors='coerce')
        max_h = hl.max()
        if max_h > 0:
            df['session_progress'] = (1.0 - hl / max_h).clip(0, 1).round(4)
        else:
            df['session_progress'] = 0.5
    else:
        df['session_progress'] = 0.5

    return df


# ── STATS DESCRIPTIVES (accumulées en streaming) ──────────────────────────────

class RunningStats:
    """Calcule mean/std/min/max en streaming sans tout garder en RAM."""
    def __init__(self):
        self.cols = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'sum2': 0.0,
                                          'min': np.inf, 'max': -np.inf})

    def update(self, df):
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            c = self.cols[col]
            c['n']    += len(s)
            c['sum']  += float(s.sum())
            c['sum2'] += float((s**2).sum())
            c['min']   = min(c['min'], float(s.min()))
            c['max']   = max(c['max'], float(s.max()))

    def get(self, col):
        c = self.cols.get(col)
        if not c or c['n'] == 0:
            return None
        n    = c['n']
        mean = c['sum'] / n
        std  = np.sqrt(max(0, c['sum2'] / n - mean**2))
        return {'mean': mean, 'std': std, 'min': c['min'], 'max': c['max'], 'n': n}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days',   type=int, default=30,
                    help='Nombre de jours à traiter (défaut: 30)')
    ap.add_argument('--export', action='store_true',
                    help='Exporter MASTER_DATASET_RL.csv (streaming)')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  BUILD RL DATASET — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    # ── 1. Découverte des fichiers ────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f][-args.days:]

    if not files:
        print("ERREUR: aucun CSV trouvé.")
        sys.exit(1)

    print(f"\n[1/3] Audit qualité — {len(files)} fichiers "
          f"({os.path.basename(files[0])} → {os.path.basename(files[-1])})")

    # ── 2. Audit streaming ────────────────────────────────────────────────────
    audits = []
    valid_files = []
    total_rows = 0

    for f in files:
        a = audit_file(f)
        audits.append(a)
        status = "WARN" if a['issues'] else "OK  "
        schema = "full" if a['has_full_schema'] else "part"
        print(f"  [{status}] {a['date']}: {a['n_rows']:>6} ticks | "
              f"{a['n_cols']}cols ({schema}) | "
              f"spread={a.get('avg_spread', 'N/A')}", end="")
        if a['issues']:
            print(f" ⚠ {'; '.join(a['issues'])}")
        else:
            print()

        if a['has_required'] and a['n_rows'] >= 1000:
            valid_files.append(f)
            total_rows += a['n_rows']

    full_schema = sum(1 for a in audits if a['has_full_schema'])
    print(f"\n  Résumé: {len(valid_files)}/{len(files)} jours valides | "
          f"{total_rows:,} ticks au total | "
          f"{full_schema} jours schéma complet (39 cols)")

    if not valid_files:
        print("ERREUR: aucun jour valide.")
        sys.exit(1)

    # ── 3. Processing + Stats (streaming) ────────────────────────────────────
    print(f"\n[2/3] Feature engineering + stats (streaming)...")
    stats = RunningStats()
    tape_counts = defaultdict(int)
    strikes_seen = set()

    if args.export:
        out_f = open(OUTPUT_CSV, 'w', newline='')
        header_written = False

    processed_rows = 0
    for i, filepath in enumerate(valid_files):
        date_str = os.path.basename(filepath).replace('dataset_', '').replace('.csv', '')
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
        except Exception as e:
            print(f"  [SKIP] {date_str}: {e}")
            continue

        # Filtrage de base
        for col in ['best_bid', 'best_ask', 'fv_theoretical']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df['best_bid'] > 0]
        df = df[df['best_ask'] > 0]
        if 'fv_theoretical' in df.columns:
            df = df[df['fv_theoretical'].between(0.01, 0.99)]

        if df.empty:
            continue

        # Feature engineering
        df = add_features(df)

        # Accumuler les stats
        stats.update(df)
        if 'last_trade_side' in df.columns:
            for v, cnt in df['last_trade_side'].value_counts().items():
                tape_counts[v] += cnt
        if 'market_strike' in df.columns:
            strikes_seen.update(df['market_strike'].dropna().unique())

        processed_rows += len(df)

        # Export streaming
        if args.export:
            if not header_written:
                df.to_csv(out_f, index=False)
                header_written = True
            else:
                df.to_csv(out_f, index=False, header=False)

        print(f"  [{i+1:2d}/{len(valid_files)}] {date_str}: {len(df):,} lignes processées")

    if args.export:
        out_f.close()
        size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
        print(f"\n  Export → {OUTPUT_CSV} ({size_mb:.1f} MB, {processed_rows:,} lignes)")

    # ── 4. Rapport de qualité RL ──────────────────────────────────────────────
    print(f"\n[3/3] Statistiques RL")
    print(f"{'─'*60}")
    print(f"  Ticks totaux traités : {processed_rows:,}")
    if strikes_seen:
        print(f"  Strikes tradés       : {sorted([int(s) for s in strikes_seen if not np.isnan(float(s))])}")
    if tape_counts:
        total_tape = sum(tape_counts.values())
        print(f"  Tape (last_trade)    : {dict(tape_counts)} — {total_tape:,} trades")

    print(f"\n  Distribution features clés (pour normalisation RL) :")
    key_features = [
        'fv_theoretical', 'spread', 'imbalance', 'vpin_proxy',
        'vol_realized_10m', 'fv_velocity', 'funding_rate',
        'bid_vol_1', 'ask_vol_1', 'hours_left',
    ]
    for col in key_features:
        s = stats.get(col)
        if s:
            print(f"  {col:<22} mean={s['mean']:>8.4f}  std={s['std']:>7.4f}  "
                  f"[{s['min']:>7.4f}, {s['max']:>7.4f}]")

    # Colonnes RL disponibles
    all_available = set(stats.cols.keys())
    rl_avail   = [c for c in RL_STATE_COLS if c in all_available]
    rl_missing = [c for c in RL_STATE_COLS if c not in all_available]

    print(f"\n  Features RL disponibles  ({len(rl_avail)})  : {rl_avail}")
    if rl_missing:
        print(f"  Features RL manquantes ({len(rl_missing)})  : {rl_missing}")

    print(f"\n{'='*60}")
    print(f"  BILAN : {processed_rows:,} obs | {len(valid_files)} jours | {len(rl_avail)} features RL")
    if not args.export:
        print(f"  Pour exporter : python3 tools/build_rl_dataset.py --export")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
