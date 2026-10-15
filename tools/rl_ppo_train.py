#!/usr/bin/env python3
"""
rl_ppo_train.py — Entraînement PPO sur données historiques Polymarket.

Architecture :
- Environnement : gymnasium-compatible, 1 épisode = 1 journée de trading
- Agent : PPO (Proximal Policy Optimization) via stable-baselines3
  Policy réseau : MLP 2 couches [64, 64] — capture les interactions non-linéaires
  que le Q-Learning linéaire ne peut pas apprendre (fv × inventory × hours_left)
- Évaluation : reward moyen sur 5 derniers jours (OOS)

Pourquoi PPO plutôt que Q-Learning linéaire ?
- Q-linéaire : Q(s,a) = θ·φ(s) — relation linéaire seulement
- PPO : politique π(a|s) = MLP(s) — capture toute relation non-linéaire
- Sur-paramétrisé mais stable grâce au clipping PPO (évite les updates trop grands)
- Standard industrie pour les problèmes de contrôle continu/discret

Usage:
    python3 tools/rl_ppo_train.py
    python3 tools/rl_ppo_train.py --days 20 --steps 200000
    python3 tools/rl_ppo_train.py --days 10 --steps 50000  # rapide
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

DATA_DIR  = "data"
MODEL_DIR = "data/rl_models"

QUOTE_SIZE    = 5.0
TICK          = 0.01
MAX_INV       = 10.0
INV_PENALTY   = 0.005
SPREAD_MAP    = [0, 2, 4, 6]   # ticks par action

# ── CHARGEMENT ────────────────────────────────────────────────────────────────

def load_episodes(n_days, oos_days=5):
    """Charge les données. Retourne (train_dfs, eval_dfs)."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f][-n_days:]

    USE_COLS = {
        'best_bid', 'best_ask', 'fv_theoretical', 'bid_vol_1', 'ask_vol_1',
        'imbalance', 'hours_left', 'last_trade_side', 'last_trade_size',
        'implied_vol', 'momentum', 'funding_rate',
    }

    all_dfs = []
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
            # Sous-échantillonner à max 2000 ticks/jour
            step = max(1, len(df) // 2000)
            df = df.iloc[::step].reset_index(drop=True)
            if len(df) >= 200:
                all_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")

    if not all_dfs:
        print("ERREUR: aucune donnée valide.")
        sys.exit(1)

    train_dfs = all_dfs[:-oos_days] if len(all_dfs) > oos_days else all_dfs
    eval_dfs  = all_dfs[-oos_days:] if len(all_dfs) > oos_days else all_dfs
    return train_dfs, eval_dfs


# ── ENVIRONNEMENT GYMNASIUM ───────────────────────────────────────────────────

class PolymarketMMEnv(gym.Env):
    """
    Environnement Market Making Polymarket — gymnasium-compatible.

    Observation (8 features normalisées) :
      [fv, spread, imbalance, inventory_norm, hours_left_norm,
       vpin_proxy, implied_vol_norm, momentum_norm]

    Action (discret, 4) :
      0 = Hold
      1 = Quote tight  (2 ticks)
      2 = Quote normal (4 ticks)
      3 = Quote wide   (6 ticks)

    Reward :
      + edge capturé sur chaque fill (fv - prix_achat ou prix_vente - fv)
      - pénalité adverse selection
      - coût de portage inventaire (0.0001/tick/share)
      - pénalité finale inventaire résiduel
    """

    metadata = {'render_modes': []}

    def __init__(self, episodes_data):
        super().__init__()
        self.episodes_data = episodes_data
        self.current_ep    = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._load_episode(0)

    def _load_episode(self, idx):
        df = self.episodes_data[idx % len(self.episodes_data)]
        self.data = df.values.astype(np.float32)
        self.cols = {c: i for i, c in enumerate(df.columns)}
        self.n_steps = len(self.data)

    def _col(self, name, default=0.0):
        idx = self.cols.get(name)
        if idx is None:
            return default
        v = float(self.data[self.t][idx])
        return v if not np.isnan(v) else default

    def _get_obs(self):
        fv    = self._col('fv_theoretical', 0.5)
        bid   = self._col('best_bid', 0.0)
        ask   = self._col('best_ask', 1.0)
        spd   = max(0.0, ask - bid)
        imb   = np.clip(self._col('imbalance', 0.0), -1, 1)
        inv_n = np.clip(self.inventory / MAX_INV, -1, 1)
        hl_n  = np.clip(self._col('hours_left', 10.0) / 20.0, 0, 1)
        vpin  = abs(imb)
        ivol  = np.clip(self._col('implied_vol', 0.5) / 1.0, 0, 1)
        mom   = np.clip(self._col('momentum', 0.0) * 100, -1, 1)

        return np.array([fv, spd, imb, inv_n, hl_n, vpin, ivol, mom],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Alterner les épisodes à chaque reset
        self._load_episode(self.current_ep)
        self.current_ep = (self.current_ep + 1) % len(self.episodes_data)

        self.t         = 0
        self.inventory = 0.0
        self.realized_pnl = 0.0
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False

        if self.t >= self.n_steps - 1:
            # Pénalité inventaire résiduel
            fv = self._col('fv_theoretical', 0.5)
            reward = -INV_PENALTY * abs(self.inventory) * QUOTE_SIZE
            if self.inventory > 0:
                reward += self.inventory * max(0, fv - 0.5) * QUOTE_SIZE * 0.3
            terminated = True
        else:
            spread_ticks = SPREAD_MAP[int(action)]

            if spread_ticks > 0:
                fv   = self._col('fv_theoretical', 0.5)
                bid  = self._col('best_bid', 0.0)
                ask  = self._col('best_ask', 1.0)
                imb  = self._col('imbalance', 0.0)

                half = spread_ticks * TICK / 2
                inv_skew = -self.inventory / MAX_INV * TICK

                our_bid = np.clip(fv - half + inv_skew, TICK, 0.99 - TICK)
                our_ask = np.clip(fv + half + inv_skew, TICK * 2, 0.99)

                # Prix suivant
                self.t += 1
                nxt_bid = self._col('best_bid', 0.0)
                nxt_ask = self._col('best_ask', 1.0)

                # BUY fill : notre bid ≥ market ask suivant
                if our_bid >= nxt_ask and self.inventory < MAX_INV:
                    self.inventory += QUOTE_SIZE
                    edge = fv - our_bid
                    adv  = max(0.0, our_bid - fv) * 0.5
                    reward += (edge - adv) * QUOTE_SIZE

                # SELL fill : notre ask ≤ market bid suivant, et on a l'inventaire
                elif our_ask <= nxt_bid and self.inventory > 0:
                    sz = min(QUOTE_SIZE, self.inventory)
                    self.inventory -= sz
                    reward += (our_ask - fv) * sz
                    self.realized_pnl += reward
            else:
                self.t += 1

            # Coût de portage
            reward -= abs(self.inventory) * 0.0001

        if self.t >= self.n_steps:
            terminated = True

        obs = self._get_obs()
        return obs, float(reward), terminated, False, {}

    def render(self):
        pass


# ── CALLBACK LOGGING ──────────────────────────────────────────────────────────

class LogCallback(BaseCallback):
    def __init__(self, log_interval=10000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards   = []

    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            ep_r = self.locals.get('rewards', [0])[0]
            self.ep_rewards.append(ep_r)

        if self.num_timesteps % self.log_interval == 0 and self.ep_rewards:
            recent = np.mean(self.ep_rewards[-50:]) if len(self.ep_rewards) >= 50 else np.mean(self.ep_rewards)
            print(f"  [{self.num_timesteps:>7} steps] reward_moy={recent:>8.4f} "
                  f"| episodes={len(self.ep_rewards)}")
        return True


# ── ÉVALUATION ────────────────────────────────────────────────────────────────

def evaluate(model, eval_dfs, n_episodes=None):
    """Évalue le modèle sur les données OOS."""
    n_episodes = n_episodes or len(eval_dfs)
    rewards, fills = [], []
    for i in range(n_episodes):
        df = eval_dfs[i % len(eval_dfs)]
        env = PolymarketMMEnv([df])
        obs, _ = env.reset()
        total_r = 0
        n_fill  = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    return np.mean(rewards), np.std(rewards)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days',  type=int, default=15,
                    help='Jours de données (défaut: 15)')
    ap.add_argument('--steps', type=int, default=100000,
                    help='Steps PPO (défaut: 100k)')
    ap.add_argument('--oos',   type=int, default=3,
                    help='Jours OOS réservés pour eval (défaut: 3)')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  PPO MARKET MAKING — stable-baselines3")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Données
    print(f"\nChargement ({args.days} jours, {args.oos} OOS)...")
    train_dfs, eval_dfs = load_episodes(args.days, args.oos)
    print(f"  Train: {len(train_dfs)} jours | Eval: {len(eval_dfs)} jours")
    print(f"  Ticks/jour (échantillonnés): "
          f"{np.mean([len(d) for d in train_dfs]):.0f}")

    # Env
    train_env = Monitor(PolymarketMMEnv(train_dfs))

    # Baseline heuristique
    print("\nBaseline heuristique (A-S statique)...")
    heur_rewards = []
    for df in eval_dfs:
        env = PolymarketMMEnv([df])
        obs, _ = env.reset()
        total = 0
        done  = False
        while not done:
            inv = obs[3]  # inventory normalisé
            action = 3 if abs(inv) > 0.8 else 2
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        heur_rewards.append(total)
    heur_mean = np.mean(heur_rewards)
    print(f"  Heuristique: {heur_mean:.4f} USDC/jour")

    # PPO
    print(f"\nEntraînement PPO ({args.steps:,} steps)...")
    print(f"  Policy: MLP [64, 64] | lr=3e-4 | batch=64 | n_steps=1024")

    model = PPO(
        policy       = "MlpPolicy",
        env          = train_env,
        learning_rate= 3e-4,
        n_steps      = 1024,
        batch_size   = 64,
        n_epochs     = 10,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_range   = 0.2,
        ent_coef     = 0.01,   # encourage l'exploration
        verbose      = 0,
        policy_kwargs= dict(net_arch=[64, 64]),
    )

    model.learn(
        total_timesteps  = args.steps,
        callback         = LogCallback(log_interval=args.steps // 10),
        progress_bar     = False,
    )

    # Sauvegarde
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = os.path.join(MODEL_DIR, f"ppo_mm_{ts}")
    model.save(model_path)
    print(f"\n  Modèle sauvegardé → {model_path}.zip")

    # Évaluation OOS
    print(f"\nÉvaluation OOS ({len(eval_dfs)} jours)...")
    rl_mean, rl_std = evaluate(model, eval_dfs)

    print(f"\n{'─'*60}")
    print(f"  RÉSULTATS FINAUX")
    print(f"{'─'*60}")
    print(f"  PPO (MLP 64x64)  : {rl_mean:>8.4f} ± {rl_std:.4f} USDC/jour")
    print(f"  Heuristique A-S  : {heur_mean:>8.4f} USDC/jour")
    delta = rl_mean - heur_mean
    print(f"  Delta PPO vs A-S : {delta:>+.4f} USDC/jour")

    if delta > 0:
        verdict = "PPO SURPASSE la heuristique"
    elif rl_mean > 0:
        verdict = "PPO positif, encore sous la heuristique — augmenter steps"
    else:
        verdict = "PPO negatif — revoir reward ou features"

    print(f"\n  VERDICT : {verdict}")
    print(f"\n  Pour continuer : python3 tools/rl_ppo_train.py --days 25 --steps 500000")
    print(f"{'='*60}\n")

    return rl_mean, heur_mean


if __name__ == '__main__':
    main()
