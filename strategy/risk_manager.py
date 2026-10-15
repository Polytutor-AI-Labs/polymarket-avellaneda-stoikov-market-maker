"""
Risk Manager pour le market maker.

Circuit breakers :
1. Max daily loss -> pull all quotes
2. Vol spike BTC (>2% en 10s) -> halt temporaire 30s
3. Temps critique avant expiry -> spread multiplier puis halt
4. API error rate -> pause si trop d'erreurs
"""

import time
from collections import deque


class RiskManager:
    def __init__(self, max_daily_loss=-10.0, vol_spike_threshold=0.02,
                 vol_spike_window=10, min_hours_to_expiry=0.5,
                 max_api_errors_per_minute=10):
        self.max_daily_loss = max_daily_loss
        self.vol_spike_threshold = vol_spike_threshold
        self.vol_spike_window = vol_spike_window
        self.min_hours_to_expiry = min_hours_to_expiry
        self.max_api_errors = max_api_errors_per_minute

        # State
        self.daily_pnl = 0.0
        self.is_halted = False
        self.halt_reason = ""
        self.halt_until = 0

        # BTC price history pour spike detection
        self.btc_prices = deque(maxlen=200)
        # API errors timestamps
        self.api_errors = deque(maxlen=100)

        # Spread adaptatif — fill rate feedback
        self.fill_timestamps = deque(maxlen=200)
        self.adaptive_mult = 1.0
        self._last_adaptive_update = 0

    def update_btc(self, price):
        self.btc_prices.append((time.time(), price))

    def record_api_error(self):
        self.api_errors.append(time.time())

    def update_pnl(self, realized_pnl, unrealized_pnl=0.0):
        # Circuit breaker sur le PnL TOTAL (réalisé + latent)
        # Avant: ignorait l'unrealized → perte de $12 sans déclencher le halt
        self.daily_pnl = realized_pnl + unrealized_pnl

    def record_fill(self):
        """Enregistre un fill pour le calcul du spread adaptatif."""
        self.fill_timestamps.append(time.time())

    def _update_adaptive_spread(self):
        """
        Ajuste le multiplicateur de spread en fonction du fill rate.
        - > 3 fills/min → trop agressif → élargir (max 2.0x)
        - 0 fills en 10 min → trop large → resserrer (min 0.7x)
        - Entre les deux → converger lentement vers 1.0
        """
        now = time.time()
        if now - self._last_adaptive_update < 30:
            return  # Update toutes les 30s max
        self._last_adaptive_update = now

        # Fill rate sur les 5 dernières minutes
        window = 300
        recent = sum(1 for t in self.fill_timestamps if now - t < window)
        fills_per_min = recent / (window / 60)

        if fills_per_min > 3.0:
            # Trop de fills → élargir
            self.adaptive_mult = min(2.0, self.adaptive_mult + 0.05)
        elif fills_per_min > 1.0:
            # Zone optimale → converger vers 1.0
            self.adaptive_mult += (1.0 - self.adaptive_mult) * 0.1
        elif recent == 0 and len(self.fill_timestamps) > 0:
            last_fill = self.fill_timestamps[-1]
            minutes_since = (now - last_fill) / 60
            if minutes_since > 10:
                # Aucun fill depuis 10 min → resserrer (floor 0.85 pour garder >2 ticks sur 3c spread)
                self.adaptive_mult = max(0.85, self.adaptive_mult - 0.03)
        else:
            # Peu de fills → converger doucement vers 1.0
            self.adaptive_mult += (1.0 - self.adaptive_mult) * 0.05

    def check(self, hours_left, inventory_abs=0):
        """
        Verifie tous les circuit breakers.

        Args:
            hours_left: Heures restantes avant expiry
            inventory_abs: Valeur absolue de l'inventory net (pour panic close)

        Returns:
            dict avec :
              halt: bool (pull all quotes immédiatement)
              spread_multiplier: float (1.0 = normal, 2.0+ = élargir)
              reason: str
              panic_close: bool (liquidation forcée d'inventory)
        """
        now = time.time()

        # Resume depuis halt temporaire
        if self.is_halted and self.halt_until > 0 and now > self.halt_until:
            self.is_halted = False
            self.halt_reason = ""
            self.halt_until = 0

        if self.is_halted:
            return {"halt": True, "spread_multiplier": 1.0, "reason": self.halt_reason, "panic_close": False}

        # 1. Daily loss limit → halt + panic close pour liquider l'inventaire
        if self.daily_pnl < self.max_daily_loss:
            self.is_halted = True
            self.halt_reason = f"Perte max atteinte ({self.daily_pnl:.2f}$)"
            return {"halt": True, "spread_multiplier": 1.0, "reason": self.halt_reason,
                    "panic_close": True, "cancel_immediate": True}

        # 2. Vol spike BTC
        btc_move = self._btc_move_pct()
        if abs(btc_move) > self.vol_spike_threshold:
            self.is_halted = True
            self.halt_until = now + 30
            self.halt_reason = f"Vol spike BTC {btc_move*100:.2f}% en {self.vol_spike_window}s"
            return {"halt": True, "spread_multiplier": 1.0, "reason": self.halt_reason, "panic_close": False}

        # 3. Temps avant expiry -> spread multiplier puis halt
        spread_mult = 1.0
        panic_close = False

        if hours_left < 2.0:
            # 2h -> 1.5x, 1h -> 2x, 0.5h -> 3x
            spread_mult = max(1.0, 3.0 / max(hours_left, 0.1))

            # PANIC CLOSE : < 2h et inventory > 20 shares → liquidation forcée
            if inventory_abs > 20:
                panic_close = True

        if hours_left < self.min_hours_to_expiry:
            self.is_halted = True
            self.halt_reason = f"Trop proche de l'expiry ({hours_left:.1f}h)"
            return {"halt": True, "spread_multiplier": 1.0, "reason": self.halt_reason, "panic_close": False}

        # 4. API error rate
        recent_errors = sum(1 for t in self.api_errors if now - t < 60)
        if recent_errors > self.max_api_errors:
            self.is_halted = True
            self.halt_until = now + 60
            self.halt_reason = f"Trop d'erreurs API ({recent_errors}/min)"
            return {"halt": True, "spread_multiplier": 1.0, "reason": self.halt_reason, "panic_close": False}

        # Intégrer le spread adaptatif (fill rate feedback)
        self._update_adaptive_spread()
        combined_mult = spread_mult * self.adaptive_mult

        return {"halt": False, "spread_multiplier": combined_mult, "reason": "",
                "panic_close": panic_close, "adaptive_mult": self.adaptive_mult}

    def _btc_move_pct(self):
        """% de mouvement BTC sur la fenêtre vol_spike_window."""
        if len(self.btc_prices) < 2:
            return 0.0
        now = time.time()
        cutoff = now - self.vol_spike_window
        old_prices = [(t, p) for t, p in self.btc_prices if t <= cutoff]
        if not old_prices:
            return 0.0
        old_price = old_prices[-1][1]
        current_price = self.btc_prices[-1][1]
        if old_price <= 0:
            return 0.0
        return (current_price - old_price) / old_price

    def force_resume(self):
        """Resume manuel depuis un halt."""
        self.is_halted = False
        self.halt_reason = ""
        self.halt_until = 0
