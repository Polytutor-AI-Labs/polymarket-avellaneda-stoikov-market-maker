"""
AS-lite informed market maker adapté pour options binaires Polymarket.

Formule clé (spec v1.0) :
  quote_center = fair_value - (net_inventory / max_inventory) * skew_max

Pourquoi AS-lite plutôt que A-S classique :
- Le calibrage Poisson est exclu (données insuffisantes sur marchés 15-min)
- La formule A-S optimale se réduit à un arrondi sur ticks de 1 cent
- Seul le concept de skew d'inventaire est conservé
- L'edge informationnel vient du BS FV (pas du market mid)
"""

import math
from dataclasses import dataclass


@dataclass
class MMQuote:
    """Un quote unique (un côté d'un token)."""
    token_id: str
    side: str        # "BUY" ou "SELL"
    price: float
    size: float
    label: str       # "YES_BID", "YES_ASK", "NO_BID", "NO_ASK"


@dataclass
class MMQuoteSet:
    """Set complet de quotes pour un marché."""
    yes_bid: MMQuote | None
    yes_ask: MMQuote | None
    no_bid: MMQuote | None
    no_ask: MMQuote | None
    reservation_price: float
    optimal_spread: float
    inventory_skew: float


class AvellanedaStoikov:
    def __init__(self, half_spread=0.03, skew_max=0.02,
                 max_inventory=50, max_gross=200,
                 size_per_quote=5.0,
                 # Paramètres legacy acceptés mais ignorés (compat)
                 gamma=None, min_spread_ticks=None, liquidation_threshold=None,
                 dynamic_spread=None, base_spread_ticks=None, min_spread_range=None):
        self.half_spread   = half_spread    # demi-spread par côté en $ (défaut 3 cents)
        self.skew_max      = skew_max       # skew max d'inventaire en $ (défaut 2 cents)
        self.max_inventory = max_inventory  # inventaire net max (shares)
        self.max_gross     = max_gross      # inventaire brut max yes+no (shares)
        self.size_per_quote = size_per_quote  # USDC par quote

    def compute_quotes(self, fair_value, inventory_net,
                       yes_token_id, no_token_id,
                       gross_inventory=0,
                       tick_size=0.01,
                       spread_multiplier=1.0,
                       yes_shares=0,       # inventaire réel YES (pour sell guard)
                       no_shares=0,        # inventaire réel NO (pour sell guard)
                       **kwargs):          # absorbe sigma, T_years, etc. (legacy)
        """
        Calcule les 4 quotes (YES bid/ask + NO bid/ask) centrés sur BS FV.

        center = fair_value - (net_inv / max_inv) * skew_max
        half   = half_spread * spread_multiplier (élargi si flow toxique ou VPIN)

        Garanties :
        - bids_yes + bids_no <= 1.00 (contrainte arbitrage)
        - asks_yes + asks_no >= 1.00 (contrainte arbitrage)
        - Pas de bids si inventaire brut >= max_gross (throttle)
        """
        fv = max(tick_size, min(1.0 - tick_size, fair_value))

        # --- Skew d'inventaire ---
        inv_ratio = max(-1.0, min(1.0,
            inventory_net / self.max_inventory if self.max_inventory > 0 else 0))
        center = fv - inv_ratio * self.skew_max
        center = max(tick_size, min(1.0 - tick_size, center))

        # --- Demi-spread (élargi sous pression d'inventaire brut) ---
        half = self.half_spread * spread_multiplier
        gross_ratio = (gross_inventory / self.max_gross
                       if self.max_gross > 0 else 0)
        if gross_ratio > 0.7:
            # +0.10 * delta au-dessus de 70% → max +3 cents à 100%
            half += (gross_ratio - 0.7) * 0.10

        # --- YES quotes ---
        yes_bid_price = self._round_tick(center - half, tick_size, "down")
        yes_ask_price = self._round_tick(center + half, tick_size, "up")
        yes_bid_price = max(tick_size, min(yes_bid_price, 1.0 - tick_size))
        yes_ask_price = max(tick_size, min(yes_ask_price, 1.0 - tick_size))
        if yes_ask_price <= yes_bid_price:
            yes_ask_price = yes_bid_price + tick_size

        # --- NO quotes (complément de YES) ---
        no_center = 1.0 - center
        no_bid_price = self._round_tick(no_center - half, tick_size, "down")
        no_ask_price = self._round_tick(no_center + half, tick_size, "up")
        no_bid_price = max(tick_size, min(no_bid_price, 1.0 - tick_size))
        no_ask_price = max(tick_size, min(no_ask_price, 1.0 - tick_size))
        if no_ask_price <= no_bid_price:
            no_ask_price = no_bid_price + tick_size

        # --- Contrainte d'arbitrage ---
        # Les bids ne peuvent pas sommer > 1.00 (sinon un arber nous drainerait)
        if yes_bid_price + no_bid_price > 1.0:
            excess = yes_bid_price + no_bid_price - 1.0
            adj = math.ceil(excess / 2 / tick_size) * tick_size
            yes_bid_price = max(tick_size, yes_bid_price - adj)
            no_bid_price  = max(tick_size, no_bid_price  - adj)
        # Les asks ne peuvent pas sommer < 1.00
        if yes_ask_price + no_ask_price < 1.0:
            deficit = 1.0 - yes_ask_price - no_ask_price
            adj = math.ceil(deficit / 2 / tick_size) * tick_size
            yes_ask_price = min(1.0 - tick_size, yes_ask_price + adj)
            no_ask_price  = min(1.0 - tick_size, no_ask_price  + adj)

        # --- Tailles (en shares = USDC / prix) ---
        yes_bid_size = self.size_per_quote / yes_bid_price if yes_bid_price > 0 else 0
        yes_ask_size = self.size_per_quote / yes_ask_price if yes_ask_price > 0 else 0
        no_bid_size  = self.size_per_quote / no_bid_price  if no_bid_price  > 0 else 0
        no_ask_size  = self.size_per_quote / no_ask_price  if no_ask_price  > 0 else 0

        # --- Limites d'inventaire ---
        # Ne pas acheter si au max net ou si inventaire brut saturé
        can_buy  = inventory_net < self.max_inventory and gross_ratio < 1.0
        # Ne pas vendre ce qu'on n'a pas (Polymarket interdit le short selling)
        can_sell_yes = inventory_net > -self.max_inventory and yes_shares >= 1
        can_sell_no  = inventory_net < self.max_inventory and no_shares >= 1

        # Sell size = min(inventaire disponible, USDC/prix) — hard guard contre over-sell
        yes_ask_size = min(yes_ask_size, float(yes_shares)) if yes_shares > 0 else yes_ask_size
        no_ask_size  = min(no_ask_size,  float(no_shares))  if no_shares  > 0 else no_ask_size

        yes_bid_q = MMQuote(yes_token_id, "BUY",  yes_bid_price, yes_bid_size, "YES_BID") if can_buy  else None
        yes_ask_q = MMQuote(yes_token_id, "SELL", yes_ask_price, yes_ask_size, "YES_ASK") if can_sell_yes else None
        no_bid_q  = MMQuote(no_token_id,  "BUY",  no_bid_price,  no_bid_size,  "NO_BID")  if can_buy  else None
        no_ask_q  = MMQuote(no_token_id,  "SELL", no_ask_price,  no_ask_size,  "NO_ASK")  if can_sell_no  else None

        return MMQuoteSet(
            yes_bid=yes_bid_q,
            yes_ask=yes_ask_q,
            no_bid=no_bid_q,
            no_ask=no_ask_q,
            reservation_price=center,
            optimal_spread=half * 2,
            inventory_skew=inv_ratio * self.skew_max,
        )

    def _round_tick(self, price, tick_size, direction="nearest"):
        if direction == "down":
            return math.floor(price / tick_size) * tick_size
        elif direction == "up":
            return math.ceil(price / tick_size) * tick_size
        return round(price / tick_size) * tick_size
