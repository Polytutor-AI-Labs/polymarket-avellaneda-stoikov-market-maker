"""
Delta binaire pour options digitales (cash-or-nothing).

Formule :
    d2 = (ln(S/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
    N'(d2) = exp(-0.5*d2^2) / sqrt(2*pi)
    delta = N'(d2) / (S * sigma * sqrt(T))   [BTC par share]

Le delta mesure combien de BTC il faut détenir pour hedger
1 share d'option binaire YES.
"""

import math


def compute_binary_delta(S, K, T_years, sigma):
    """
    Delta unitaire d'une option binaire YES (BTC par share).

    Args:
        S: prix spot BTC
        K: strike de l'option
        T_years: temps restant en années
        sigma: volatilité annualisée

    Returns:
        float delta en BTC par share YES (toujours >= 0)
    """
    # Cas dégénérés : pas de temps ou vol nulle → delta = 0
    if T_years <= 1e-8 or sigma <= 0 or S <= 0:
        return 0.0

    sqrt_T = math.sqrt(T_years)
    d2 = (math.log(S / K) - 0.5 * sigma**2 * T_years) / (sigma * sqrt_T)

    # PDF de la normale standard : N'(d2) = exp(-0.5*d2^2) / sqrt(2*pi)
    nprime_d2 = math.exp(-0.5 * d2**2) / math.sqrt(2.0 * math.pi)

    # Delta = N'(d2) / (S * sigma * sqrt(T))
    delta = nprime_d2 / (S * sigma * sqrt_T)

    return delta


def compute_portfolio_delta(markets, btc_price, implied_vol):
    """
    Delta net agrégé du portefeuille en BTC.

    Pour chaque marché :
        delta_contribution = delta_unitaire * (yes_shares - no_shares)
    Le delta total est la somme sur tous les marchés.

    Args:
        markets: liste de MarketContext avec strike, expiry, yes_shares, no_shares
        btc_price: prix spot BTC
        implied_vol: volatilité annualisée

    Returns:
        dict avec :
            - total_delta_btc: delta net en BTC
            - total_delta_usd: delta net en USD
            - per_market: liste de dicts par marché
    """
    from datetime import datetime, timezone

    per_market = []
    total_delta_btc = 0.0

    for ctx in markets:
        # Calcul du temps restant en années
        try:
            clean_expiry = ctx.expiry.replace("Z", "")
            expiry_dt = datetime.fromisoformat(clean_expiry)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            T_years = max(0, (expiry_dt - now).total_seconds() / (365 * 24 * 3600))
        except Exception:
            T_years = 0.0

        delta_unit = compute_binary_delta(btc_price, ctx.strike, T_years, implied_vol)
        net_inventory = ctx.yes_shares - ctx.no_shares
        delta_btc = delta_unit * net_inventory

        per_market.append({
            "strike": ctx.strike,
            "strike_label": ctx.strike_label,
            "delta_unit": delta_unit,
            "net_inventory": net_inventory,
            "delta_btc": delta_btc,
            "T_years": T_years,
        })

        total_delta_btc += delta_btc

    return {
        "total_delta_btc": total_delta_btc,
        "total_delta_usd": total_delta_btc * btc_price,
        "per_market": per_market,
    }
