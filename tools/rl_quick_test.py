#!/usr/bin/env python3
"""
rl_quick_test.py — Test rapide RL market making sur données historiques.

Implémentation pure numpy (pas de torch requis) :
- Environnement gym-compatible : état du carnet d'ordres + inventaire
- Agent : Q-Learning avec approximation linéaire (Linear FA)
- Baseline : agent aléatoire + agent A-S statique (pour comparaison)

État (6 features) :
  [fv, spread, imbalance, inventory_norm, hours_left_norm, vpin_proxy]

Actions (4) :
  0 = Hold (ne rien faire)
  1 = Quote tight  (spread = 2 ticks)
  2 = Quote normal (spread = 4 ticks)
  3 = Quote wide   (spread = 6 ticks)

Reward :
  + spread capturé sur chaque fill
  - pénalité d'inventaire (position non liquidée)
  - pénalité adverse selection (fill dans le mauvais sens)

Usage:
    python3 tools/rl_quick_test.py
    python3 tools/rl_quick_test.py --days 5 --episodes 200
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"

# ── HYPERPARAMÈTRES ───────────────────────────────────────────────────────────
QUOTE_SIZE     = 5.0       # USDC par quote
TICK           = 0.01      # Tick Polymarket
MAX_INV        = 10.0      # Inventaire max en shares
INV_PENALTY    = 0.005     # Pénalité par share d'inventaire à la fin
GAMMA_RL       = 0.99      # Discount factor (marché intraday)

# Q-Learning linéaire
ALPHA          = 0.01      # Learning rate
EPSILON_START  = 1.0       # Exploration initiale
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.995

N_ACTIONS      = 4
SPREAD_BY_ACTION = [0, 2, 4, 6]  # ticks pour chaque action (0=hold)

# ── CHARGEMENT DONNÉES ────────────────────────────────────────────────────────

def load_episodes(n_days=5):
    """Charge N jours de données. Retourne liste de DataFrames (1 par jour)."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_2026-*.csv')))
    files = [f for f in files if 'MASTER' not in f][-n_days:]

    episodes = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False,
                             usecols=lambda c: c in {
                                 'timestamp', 'btc_price', 'best_bid', 'best_ask',
                                 'bid_vol_1', 'ask_vol_1', 'imbalance',
                                 'fv_theoretical', 'hours_left',
                                 'last_trade_side', 'last_trade_size',
                             })
            for col in ['best_bid', 'best_ask', 'fv_theoretical', 'bid_vol_1',
                        'ask_vol_1', 'imbalance', 'hours_left']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['best_bid', 'best_ask', 'fv_theoretical'])
            df = df[df['best_bid'] > 0]
            df = df[df['fv_theoretical'].between(0.05, 0.95)]
            df = df[df['hours_left'] > 0.5] if 'hours_left' in df.columns else df

            if len(df) > 500:
                # Sous-échantillonner à max 3000 ticks/jour pour la vitesse
                step = max(1, len(df) // 3000)
                df = df.iloc[::step].reset_index(drop=True)
                episodes.append(df)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(f)}: {e}")

    return episodes


# ── ENVIRONNEMENT ─────────────────────────────────────────────────────────────

class MarketMakingEnv:
    """
    Environnement de market making sur options binaires Polymarket.

    Un épisode = une journée de trading.
    À chaque step : l'agent choisit son spread, on simule les fills,
    on calcule le reward.
    """

    def __init__(self, df):
        self.data = df.values  # numpy pour la vitesse
        self.cols = {c: i for i, c in enumerate(df.columns)}
        self.n_steps = len(df)

    def reset(self):
        self.t = 0
        self.inventory = 0.0
        self.realized_pnl = 0.0
        self.n_fills = 0
        return self._get_state()

    def _get_state(self):
        if self.t >= self.n_steps:
            return np.zeros(6)
        row = self.data[self.t]

        fv  = row[self.cols.get('fv_theoretical', 0)] if 'fv_theoretical' in self.cols else 0.5
        bid = row[self.cols.get('best_bid', 1)]       if 'best_bid' in self.cols else 0.0
        ask = row[self.cols.get('best_ask', 2)]       if 'best_ask' in self.cols else 1.0
        spread = max(0, ask - bid)

        imb = row[self.cols.get('imbalance', 3)] if 'imbalance' in self.cols else 0.0
        imb = np.clip(imb, -1, 1)

        inv_norm = np.clip(self.inventory / MAX_INV, -1, 1)

        hl = row[self.cols.get('hours_left', 4)] if 'hours_left' in self.cols else 10.0
        hl_norm = np.clip(hl / 20.0, 0, 1)

        # VPIN proxy simple : |imbalance| comme proxy de toxicité
        vpin = abs(imb)

        return np.array([fv, spread, imb, inv_norm, hl_norm, vpin], dtype=np.float32)

    def step(self, action):
        """
        Exécute une action. Retourne (next_state, reward, done).
        """
        if self.t >= self.n_steps - 1:
            # Fin d'épisode : pénalité inventaire résiduel
            final_row = self.data[-1]
            last_fv = final_row[self.cols.get('fv_theoretical', 0)] if 'fv_theoretical' in self.cols else 0.5
            reward = -INV_PENALTY * abs(self.inventory) * QUOTE_SIZE
            # Liquider au FV (mark-to-market)
            if self.inventory > 0:
                reward += self.inventory * (last_fv - 0.5) * QUOTE_SIZE * 0.5
            self.t += 1
            return self._get_state(), reward, True

        row = self.data[self.t]
        reward = 0.0

        spread_ticks = SPREAD_BY_ACTION[action]

        if spread_ticks > 0 and 'fv_theoretical' in self.cols:
            fv     = row[self.cols['fv_theoretical']]
            bid_m  = row[self.cols['best_bid']] if 'best_bid' in self.cols else 0.0
            ask_m  = row[self.cols['best_ask']] if 'best_ask' in self.cols else 1.0
            imb    = row[self.cols['imbalance']] if 'imbalance' in self.cols else 0.0

            half_sp = spread_ticks * TICK / 2

            # Skew inventaire (réduction spread côté long si on a inventory)
            inv_skew = -self.inventory / MAX_INV * TICK
            our_bid = fv - half_sp + inv_skew
            our_ask = fv + half_sp + inv_skew

            our_bid = max(TICK, min(our_bid, 0.99 - TICK))
            our_ask = max(TICK + TICK, min(our_ask, 0.99))

            # Simulation fill : notre BID se remplit si market_ask tombe en dessous
            next_t = min(self.t + 1, self.n_steps - 1)
            next_row = self.data[next_t]
            next_ask = next_row[self.cols['best_ask']] if 'best_ask' in self.cols else 1.0
            next_bid = next_row[self.cols['best_bid']] if 'best_bid' in self.cols else 0.0

            # BUY fill
            if our_bid >= next_ask and self.inventory < MAX_INV:
                self.inventory += QUOTE_SIZE
                # Reward = demi-spread capturé - coût adverse selection estimé
                edge = fv - our_bid
                adv_sel = max(0, our_bid - fv) * 0.5  # si on achète au-dessus FV
                reward += (edge - adv_sel) * QUOTE_SIZE
                self.n_fills += 1

            # SELL fill (seulement si on a de l'inventaire)
            elif our_ask <= next_bid and self.inventory > 0:
                sell_size = min(QUOTE_SIZE, self.inventory)
                self.inventory -= sell_size
                edge = our_ask - fv
                reward += edge * sell_size
                self.n_fills += 1
                self.realized_pnl += reward

            # Pénalité inventaire continue (cost of carry)
            reward -= abs(self.inventory) * 0.0001

        self.t += 1
        done = (self.t >= self.n_steps)
        return self._get_state(), reward, done


# ── AGENT Q-LEARNING LINÉAIRE ─────────────────────────────────────────────────

class LinearQAgent:
    """
    Q-Learning avec approximation linéaire.
    Q(s, a) = theta[a] · phi(s)  où phi(s) est le vecteur d'état augmenté.

    Avantages vs réseau de neurones :
    - Convergence garantie (sous conditions)
    - Entraînement en quelques secondes sur CPU
    - Interprétable : theta[a] montre l'importance de chaque feature
    - Pas de torch requis
    """

    def __init__(self, n_features=6, n_actions=N_ACTIONS):
        self.n_features = n_features
        self.n_actions  = n_actions
        # Initialisation proche de 0 (évite biais initial)
        self.theta = np.zeros((n_actions, n_features))
        self.epsilon = EPSILON_START

    def _phi(self, state):
        """Feature vector (identité ici — possible d'ajouter des features poly)."""
        return state.astype(np.float64)

    def q_values(self, state):
        phi = self._phi(state)
        return self.theta @ phi  # (n_actions,)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def update(self, state, action, reward, next_state, done):
        phi      = self._phi(state)
        phi_next = self._phi(next_state)

        q_current = self.theta[action] @ phi
        if done:
            td_target = reward
        else:
            td_target = reward + GAMMA_RL * np.max(self.theta @ phi_next)

        td_error = td_target - q_current
        self.theta[action] += ALPHA * td_error * phi

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ── BASELINES ─────────────────────────────────────────────────────────────────

def run_episode_random(env):
    """Agent aléatoire — baseline inférieure."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.random.randint(N_ACTIONS)
        state, reward, done = env.step(action)
        total_reward += reward
    return total_reward, env.n_fills


def run_episode_heuristic(env):
    """
    Agent heuristique A-S simplifié : quote normal (action=2) toujours,
    avec pull si inventory trop élevé.
    """
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        inv_norm = state[3]
        # Pull si inventaire > 80% de max
        action = 3 if abs(inv_norm) > 0.8 else 2
        state, reward, done = env.step(action)
        total_reward += reward
    return total_reward, env.n_fills


def run_episode_agent(agent, env, train=True):
    """Agent RL — entraîne si train=True."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
    if train:
        agent.decay_epsilon()
    return total_reward, env.n_fills


# ── ENTRAÎNEMENT ──────────────────────────────────────────────────────────────

def train(episodes_data, n_train_episodes=200, n_eval_episodes=20):
    """
    Entraîne l'agent RL et compare aux baselines.
    """
    n_days = len(episodes_data)
    agent  = LinearQAgent()

    print(f"  {n_days} jours de données | {n_train_episodes} épisodes d'entraînement")
    print(f"  Agent: Q-Learning linéaire | State: 6 features | Actions: {N_ACTIONS}")

    # ── Baselines (évaluation pré-entraînement) ──
    rand_rewards, heur_rewards = [], []
    for df in episodes_data:
        env = MarketMakingEnv(df)
        r, _ = run_episode_random(env)
        rand_rewards.append(r)
        r, _ = run_episode_heuristic(env)
        heur_rewards.append(r)

    print(f"\n  Baselines (sur {n_days} jours) :")
    print(f"    Agent aléatoire     : {np.mean(rand_rewards):>8.4f} USDC/jour")
    print(f"    Heuristique A-S     : {np.mean(heur_rewards):>8.4f} USDC/jour")

    # ── Entraînement ──
    rewards_history = []
    window = 20

    print(f"\n  Entraînement... (epsilon: {agent.epsilon:.2f} → {EPSILON_MIN:.2f})")
    for ep in range(n_train_episodes):
        df = episodes_data[ep % n_days]
        env = MarketMakingEnv(df)
        reward, fills = run_episode_agent(agent, env, train=True)
        rewards_history.append(reward)

        if (ep + 1) % 50 == 0:
            recent = np.mean(rewards_history[-window:])
            print(f"    Ep {ep+1:>3}/{n_train_episodes}  "
                  f"reward_moy={recent:>8.4f}  "
                  f"epsilon={agent.epsilon:.3f}  "
                  f"fills_last={fills}")

    # ── Évaluation finale ──
    eval_rewards, eval_fills = [], []
    for df in episodes_data:
        env = MarketMakingEnv(df)
        r, f = run_episode_agent(agent, env, train=False)
        eval_rewards.append(r)
        eval_fills.append(f)

    rl_mean = np.mean(eval_rewards)

    print(f"\n  Résultats finaux (évaluation greedy) :")
    print(f"    RL (LinearQ)        : {rl_mean:>8.4f} USDC/jour "
          f"({np.mean(eval_fills):.1f} fills/jour)")
    print(f"    Heuristique A-S     : {np.mean(heur_rewards):>8.4f} USDC/jour")
    print(f"    Agent aléatoire     : {np.mean(rand_rewards):>8.4f} USDC/jour")

    improvement = rl_mean - np.mean(heur_rewards)
    print(f"\n  Delta RL vs heuristique : {improvement:>+.4f} USDC/jour")

    # ── Poids appris (interprétabilité) ──
    feature_names = ['fv', 'spread', 'imbalance', 'inventory', 'hours_left', 'vpin']
    action_names  = ['Hold', 'Tight(2t)', 'Normal(4t)', 'Wide(6t)']
    print(f"\n  Poids appris θ (interprétabilité) :")
    print(f"  {'Action':<14}", end="")
    for fn in feature_names:
        print(f"{fn:>10}", end="")
    print()
    for i, an in enumerate(action_names):
        print(f"  {an:<14}", end="")
        for w in agent.theta[i]:
            print(f"{w:>10.4f}", end="")
        print()

    return agent, rl_mean, np.mean(heur_rewards)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days',     type=int, default=5,
                    help='Jours de données à charger (défaut: 5)')
    ap.add_argument('--episodes', type=int, default=200,
                    help='Épisodes d\'entraînement (défaut: 200)')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  RL QUICK TEST — Q-Learning Linéaire")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    print(f"\nChargement des données ({args.days} derniers jours)...")
    episodes_data = load_episodes(args.days)

    if not episodes_data:
        print("ERREUR: aucune donnée trouvée.")
        sys.exit(1)

    total_ticks = sum(len(d) for d in episodes_data)
    print(f"  {len(episodes_data)} jours | {total_ticks:,} ticks")

    agent, rl_score, baseline_score = train(
        episodes_data,
        n_train_episodes=args.episodes,
    )

    print(f"\n{'='*60}")
    if rl_score > baseline_score:
        verdict = "RL SURPASSE la heuristique — signal positif"
    elif rl_score > 0:
        verdict = "RL POSITIF mais sous la heuristique — continuer à entraîner"
    else:
        verdict = "RL NEGATIF — revoir reward function ou features"
    print(f"  VERDICT : {verdict}")
    print(f"\n  Prochaines étapes si signal positif :")
    print(f"  1. Augmenter à --days 20 --episodes 1000 (attention RAM)")
    print(f"  2. Passer à PPO (stable-baselines3) pour une politique non-linéaire")
    print(f"  3. Ajouter features manquantes (vol_realized, funding_rate)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
