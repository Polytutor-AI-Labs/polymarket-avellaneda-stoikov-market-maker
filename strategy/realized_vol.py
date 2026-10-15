"""
Realized Volatility Calculator

Calcule la volatilité réalisée (realized vol) sur une fenêtre glissante de ticks Binance,
puis blend avec l'IV Deribit pour obtenir une estimation plus réactive.

Utilise EWMA (Exponentially Weighted Moving Average) pour donner plus de poids aux données récentes.
"""

import math
from collections import deque
import time


class RealizedVolCalculator:
    def __init__(self, window_minutes=60, blend_weight_realized=0.3):
        """
        Args:
            window_minutes: Taille de la fenêtre glissante en minutes
            blend_weight_realized: Poids de la realized vol dans le blend (0.3 = 30% realized, 70% IV Deribit)
        """
        self.window_minutes = window_minutes
        self.blend_weight = blend_weight_realized

        # Store (timestamp, price)
        self.price_ticks = deque(maxlen=10000)  # ~16h de ticks à 100ms

        # Cache
        self._cached_realized_vol = None
        self._last_calc_time = 0
        self._calc_interval = 10  # Recalculer toutes les 10s

    def add_price(self, price):
        """Ajoute un tick de prix BTC."""
        self.price_ticks.append((time.time(), price))

    def get_realized_vol(self):
        """
        Retourne la vol réalisée annualisée.

        Returns:
            float: Volatilité annualisée (ex: 0.65 = 65%)
            None si pas assez de données
        """
        now = time.time()

        # Use cache si récent
        if self._cached_realized_vol is not None and now - self._last_calc_time < self._calc_interval:
            return self._cached_realized_vol

        if len(self.price_ticks) < 100:  # Minimum 100 ticks (~10 secondes)
            return None

        # Filtrer sur la fenêtre
        cutoff = now - self.window_minutes * 60
        window_ticks = [(t, p) for t, p in self.price_ticks if t >= cutoff]

        if len(window_ticks) < 100:
            return None

        # Calculer les log returns
        prices = [p for _, p in window_ticks]
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = math.log(prices[i] / prices[i-1])
                returns.append(ret)

        if len(returns) < 50:
            return None

        # Volatilité = std dev des returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        # Annualiser
        # std_dev est la vol par observation. Il faut multiplier par sqrt(N_obs_per_year).
        # N_obs_per_year = returns_per_min * minutes_per_year
        duration_minutes = (window_ticks[-1][0] - window_ticks[0][0]) / 60
        if duration_minutes <= 0:
            return None

        returns_per_min = len(returns) / duration_minutes
        minutes_per_year = 365 * 24 * 60
        observations_per_year = returns_per_min * minutes_per_year
        annualization_factor = math.sqrt(observations_per_year)

        realized_vol = std_dev * annualization_factor

        # Cache
        self._cached_realized_vol = realized_vol
        self._last_calc_time = now

        return realized_vol

    def blend_with_iv(self, deribit_iv):
        """
        Blend realized vol avec IV Deribit.

        Args:
            deribit_iv: IV de Deribit (ex: 0.70 = 70%)

        Returns:
            float: Vol blendée
        """
        realized = self.get_realized_vol()

        if realized is None:
            return deribit_iv

        # Blend : weight_realized * realized + (1 - weight_realized) * deribit
        blended = self.blend_weight * realized + (1 - self.blend_weight) * deribit_iv

        # Sanity check : cap entre 0.20 (20%) et 3.0 (300%)
        blended = max(0.20, min(3.0, blended))

        return blended

    def get_stats(self):
        """Retourne des stats pour debugging."""
        realized = self.get_realized_vol()
        return {
            "ticks_count": len(self.price_ticks),
            "window_minutes": self.window_minutes,
            "realized_vol": realized,
            "blend_weight": self.blend_weight,
        }


class SigmaKalmanFilter:
    """
    Filtre de Kalman 1D pour l'estimation de la volatilité implicite.

    Fusionne de façon optimale (bayésienne) deux sources de sigma :
      - Deribit IV    : précise mais peut être stale ou biaisée
      - Realized vol  : réactive mais bruitée (variance élevée)

    Modèle:
      State   x_k = sigma_k  (volatilité vraie, inobservable)
      Process x_{k+1} = x_k + w_k,   w_k ~ N(0, Q)   sigma dérive lentement
      Obs (D) y_d = x_k + v_d,        v_d ~ N(0, R_d)  Deribit IV (précis)
      Obs (R) y_r = x_k + v_r,        v_r ~ N(0, R_r)  Realized vol (bruitée)

    Références : Welch & Bishop (2006) "An Introduction to the Kalman Filter".
    """

    def __init__(
        self,
        initial_sigma: float = 0.65,
        process_noise: float = 0.001,
        deribit_noise: float = 0.015,
        realized_noise: float = 0.060,
    ):
        """
        Args:
            initial_sigma   : estimation initiale de sigma (ex: 0.65 = 65%)
            process_noise   : Q — à quel point sigma peut dériver entre deux updates
                              (petit = sigma change lentement)
            deribit_noise   : R_d — incertitude sur la mesure Deribit IV
                              (petit = on fait très confiance à Deribit)
            realized_noise  : R_r — incertitude sur la realized vol
                              (grand = realized vol est bruitée)
        """
        self.Q = process_noise
        self.R_d = deribit_noise
        self.R_r = realized_noise

        # État Kalman
        self._x = initial_sigma      # Estimation courante de sigma
        self._P = 0.01               # Variance de l'erreur d'estimation (incertitude)

        # Historique pour stats
        self._update_count = 0
        self._last_deribit_iv = None
        self._last_realized_vol = None

    def update(self, deribit_iv: float | None, realized_vol: float | None) -> float:
        """
        Met à jour l'estimation de sigma avec de nouvelles observations.

        Appelé à chaque fois qu'une observation est disponible (ex: toutes les 10s).

        Args:
            deribit_iv   : IV Deribit interpolée au strike (None si indisponible)
            realized_vol : vol réalisée calculée (None si pas assez de données)

        Returns:
            float : estimation courante de sigma (toujours valide)
        """
        # --- Étape Prédiction ---
        # La variance augmente légèrement (sigma peut avoir dérivé)
        P_pred = self._P + self.Q

        # --- Étape Correction : Deribit IV ---
        if deribit_iv is not None and 0.10 < deribit_iv < 3.0:
            self._last_deribit_iv = deribit_iv
            K_d = P_pred / (P_pred + self.R_d)
            self._x = self._x + K_d * (deribit_iv - self._x)
            P_pred = (1.0 - K_d) * P_pred

        # --- Étape Correction : Realized Vol ---
        if realized_vol is not None and 0.10 < realized_vol < 3.0:
            self._last_realized_vol = realized_vol
            K_r = P_pred / (P_pred + self.R_r)
            self._x = self._x + K_r * (realized_vol - self._x)
            P_pred = (1.0 - K_r) * P_pred

        # Sauvegarder la variance mise à jour
        self._P = P_pred
        self._update_count += 1

        # Borne de sécurité
        self._x = max(0.15, min(3.0, self._x))

        return self._x

    @property
    def sigma(self) -> float:
        """Estimation courante de sigma (sans mise à jour)."""
        return self._x

    @property
    def uncertainty(self) -> float:
        """Variance de l'erreur d'estimation (confiance dans l'estimation)."""
        return self._P

    def get_stats(self) -> dict:
        return {
            "sigma": round(self._x, 4),
            "uncertainty": round(self._P, 6),
            "updates": self._update_count,
            "last_deribit_iv": round(self._last_deribit_iv, 4) if self._last_deribit_iv else None,
            "last_realized_vol": round(self._last_realized_vol, 4) if self._last_realized_vol else None,
        }
