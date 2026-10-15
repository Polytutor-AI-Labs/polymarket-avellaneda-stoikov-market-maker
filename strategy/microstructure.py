"""
Signaux de microstructure du carnet d'ordres.
Analyse l'imbalance, le momentum BTC et le spread pour ajuster
dynamiquement le seuil d'edge du bot.
"""
from collections import deque
import time


class MicrostructureAnalyzer:
    def __init__(self, price_window=60):
        """
        Args:
            price_window: secondes d'historique BTC (default 60s)
        """
        # ~10 ticks/s depuis Binance, +10s buffer pour couvrir la fenêtre 60s
        self.btc_history = deque(maxlen=(price_window + 10) * 10)

    def update_btc(self, price):
        """Appelé à chaque tick Binance."""
        self.btc_history.append((time.time(), price))

    def _get_momentum(self, window_seconds):
        """Calcule le momentum BTC sur une fenêtre temporelle précise.

        Scanne la deque pour trouver le prix à (now - window_seconds).
        Plus robuste que l'approche par index [0] car indépendant du
        taux de remplissage de la deque.

        Args:
            window_seconds: fenêtre de lookback en secondes

        Returns:
            float: variation fractionnelle du prix BTC (ex: 0.001 = +0.1%)
        """
        if len(self.btc_history) < 2:
            return 0.0

        now_ts = self.btc_history[-1][0]
        target_time = now_ts - window_seconds

        # Pas assez d'historique pour cette fenêtre
        if self.btc_history[0][0] > target_time:
            return 0.0

        # Trouver le premier prix >= target_time
        old_price = None
        for ts, price in self.btc_history:
            if ts >= target_time:
                old_price = price
                break

        if old_price is None or old_price <= 0:
            return 0.0

        recent_price = self.btc_history[-1][1]
        return (recent_price - old_price) / old_price

    def compute_signals(self, bid_vol, ask_vol, spread, btc_price, strike, side="YES"):
        """
        Calcule les signaux de microstructure et un ajustement d'edge.

        Args:
            bid_vol: volume au best bid
            ask_vol: volume au best ask
            spread: ask - bid
            btc_price: prix BTC actuel
            strike: strike du marché Polymarket
            side: "YES" ou "NO" — pour orienter le momentum

        Returns:
            dict avec imbalance, momentum, spread_score, edge_adjustment
        """
        # 1. Imbalance [-1, +1] : positif = pression acheteuse sur Poly
        imbalance = 0.0
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        # 2. Momentum BTC multi-fenêtre (time-based)
        momentum_short = self._get_momentum(10)   # 10s : spikes immédiats
        momentum = self._get_momentum(30)          # 30s : signal principal
        momentum_long = self._get_momentum(60)     # 60s : trend

        # 3. Spread score [0, 1] : spread serré = meilleur environnement
        # 0 si spread >= 5 cents, 1 si spread = 0
        spread_score = max(0.0, 1.0 - spread / 0.05)

        # 4. Momentum directionnel pondéré multi-fenêtre
        # YES side : momentum positif (BTC monte vers/au-dessus du strike) = favorable
        # NO side : momentum négatif (BTC descend sous le strike) = favorable
        weighted_momentum = momentum_short * 0.4 + momentum * 0.4 + momentum_long * 0.2
        if side == "YES":
            momentum_favor = weighted_momentum
        else:
            momentum_favor = -weighted_momentum

        # 5. Composite : ajustement de l'edge
        # Conditions favorables → on ajoute à l'edge (facilite le déclenchement)
        # Conditions défavorables → on retire de l'edge (prudence)
        # Poids recalibrés (momentum renforcé — données 2026-02 montrent stdev=0.067%) :
        #   imbalance ±1 → ±0.003, momentum ±0.001 → ±0.003, spread 0-1 → +0.001
        edge_adjustment = (
            imbalance * 0.003          # imbalance forte = +/- 0.3 cents
            + spread_score * 0.001     # spread serré = + 0.1 cents
            + momentum_favor * 3.0     # momentum 0.001 (0.1%) = +/- 0.3 cents
        )

        return {
            "imbalance": round(imbalance, 4),
            "momentum": round(momentum, 6),
            "momentum_short": round(momentum_short, 6),
            "momentum_long": round(momentum_long, 6),
            "spread_score": round(spread_score, 4),
            "edge_adjustment": round(edge_adjustment, 6),
        }


class VPINCalculator:
    """
    VPIN — détecte les traders informés via le déséquilibre buy/sell.

    VPIN = |V_buy - V_sell| / (V_buy + V_sell) sur une fenêtre de N trades.
    VPIN proche de 0 = flux équilibré (noise traders).
    VPIN proche de 1 = flux unilatéral (traders informés).

    Source: Easley, de Prado & O'Hara (2012) "Flow Toxicity and Liquidity".
    Adapté pour options binaires Polymarket : side="BUY" = trade achète YES.
    """

    def __init__(self, window_trades=50, toxicity_threshold=0.35, penalty_factor=1.5):
        """
        Args:
            window_trades: nombre de trades dans la fenêtre glissante
            toxicity_threshold: VPIN au-dessus duquel le flow est considéré toxique
            penalty_factor: multiplicateur de spread max quand VPIN = 1.0
                            (interpolation linéaire entre 1.0 et penalty_factor)
        """
        self.window_trades = window_trades
        self.toxicity_threshold = toxicity_threshold
        self.penalty_factor = penalty_factor

        # Fenêtre glissante de (volume, side) — side: "BUY" ou "SELL"
        self._trades = deque(maxlen=window_trades)
        self._last_vpin = 0.0

    def add_trade(self, price: float, size: float, side: str):
        """
        Enregistre un trade.

        Args:
            price: prix du trade (0-1)
            size: volume en shares
            side: "BUY" (achat YES, lift l'ask) ou "SELL" (vente YES, hit le bid)
        """
        if size <= 0 or not side:
            return
        self._trades.append((size, side.upper()))

    def get_vpin(self) -> float:
        """
        Calcule le VPIN courant.

        Returns:
            float [0, 1] : 0 = flux équilibré, 1 = flux unilatéral (toxique).
            Retourne 0.0 si moins de 10 trades dans la fenêtre.
        """
        if len(self._trades) < 10:
            return 0.0

        buy_vol = sum(sz for sz, side in self._trades if side == "BUY")
        sell_vol = sum(sz for sz, side in self._trades if side == "SELL")
        total_vol = buy_vol + sell_vol

        if total_vol <= 0:
            return 0.0

        vpin = abs(buy_vol - sell_vol) / total_vol
        self._last_vpin = vpin
        return vpin

    def get_spread_penalty(self) -> float:
        """
        Retourne un multiplicateur de spread basé sur le VPIN.

        Returns:
            float >= 1.0 : 1.0 si flow sain, jusqu'à penalty_factor si très toxique.
        """
        vpin = self.get_vpin()

        if vpin <= self.toxicity_threshold:
            return 1.0

        # Interpolation linéaire : threshold → 1.0, 1.0 → penalty_factor
        t = (vpin - self.toxicity_threshold) / (1.0 - self.toxicity_threshold)
        penalty = 1.0 + t * (self.penalty_factor - 1.0)
        return round(penalty, 3)

    def get_stats(self) -> dict:
        """Stats pour logging et debug."""
        vpin = self.get_vpin()
        buy_vol = sum(sz for sz, side in self._trades if side == "BUY")
        sell_vol = sum(sz for sz, side in self._trades if side == "SELL")
        return {
            "vpin": round(vpin, 4),
            "n_trades": len(self._trades),
            "buy_vol": round(buy_vol, 2),
            "sell_vol": round(sell_vol, 2),
            "spread_penalty": self.get_spread_penalty(),
            "toxic": vpin > self.toxicity_threshold,
        }
