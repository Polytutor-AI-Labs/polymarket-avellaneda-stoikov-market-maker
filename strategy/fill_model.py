"""
Modeles de probabilite de fill pour le simulateur paper trading.

Extraits de analysis/mm_backtest.py pour etre reutilises par
le QuoteManager en mode SIM et le backtester.
"""

import math


def fill_probability_v1(our_size, market_vol, imbalance, is_buy):
    """Probabilite de fill basee sur volume disponible et imbalance (sigmoid simple).

    Args:
        our_size: Taille de notre ordre en shares
        market_vol: Volume au meilleur niveau (ask_vol pour buy, bid_vol pour sell)
        imbalance: Orderbook imbalance [-1, +1]
        is_buy: True si achat (notre bid fill)

    Returns:
        float: probabilite [0.05, 0.95]
    """
    if market_vol <= 0:
        return 0.05
    size_ratio = min(our_size / max(market_vol, 1.0), 3.0)
    base_prob = 1.0 / (1.0 + size_ratio)
    imb_adj = (-imbalance if is_buy else imbalance) * 0.15
    return max(0.05, min(0.95, base_prob + imb_adj))


def fill_probability_v2(our_size, market_vol, imbalance, is_buy,
                         depth_vol_2=0.0, depth_vol_3=0.0,
                         spread=0.02, realized_vol=None):
    """Modele de fill ameliore utilisant la profondeur du book.

    Ameliorations vs v1:
    1. Utilise 3 niveaux de profondeur (pas juste level 1)
    2. Spread bonus : spread serre = plus de fills
    3. Imbalance coefficient renforce (0.20 vs 0.15)
    4. Vol bonus : haute vol = plus d'ordres market = plus de fills
    5. Cap conservateur a 0.90 (vs 0.95)
    """
    if market_vol <= 0:
        return 0.05

    # 1. Base: sigmoid avec profondeur totale (3 niveaux)
    total_depth = market_vol + depth_vol_2 + depth_vol_3
    size_ratio = min(our_size / max(total_depth, 1.0), 3.0)
    base_prob = 1.0 / (1.0 + size_ratio * 1.5)

    # 2. Spread proximity: spread serre = plus probable d'etre touche
    spread_bonus = max(0.0, 0.15 * (1.0 - spread / 0.05))

    # 3. Imbalance: coefficient renforce
    imb_adj = (-imbalance if is_buy else imbalance) * 0.20

    # 4. Volatilite: haute vol = plus d'ordres market = plus de fills
    vol_bonus = 0.0
    if realized_vol is not None and not math.isnan(realized_vol) and realized_vol > 0:
        vol_bonus = max(0.0, min(0.10, (realized_vol - 0.40) * 0.5))

    prob = base_prob + spread_bonus + imb_adj + vol_bonus
    return max(0.05, min(0.90, prob))
