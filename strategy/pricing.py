import math
from datetime import datetime, timezone


# --- SURFACE DE VOLATILITÉ ---

def interpolate_iv(vol_surface, target_strike):
    """
    Interpole linéairement l'IV depuis la surface de vol Deribit.

    Args:
        vol_surface: dict {strike: iv} ex: {66000: 0.71, 68000: 0.69}
        target_strike: strike Polymarket à pricer

    Returns:
        float IV interpolée, ou None si surface vide
    """
    if not vol_surface:
        return None

    strikes = sorted(vol_surface.keys())

    # Match exact
    if target_strike in vol_surface:
        return vol_surface[target_strike]

    # Encadrer le strike cible
    lower = [s for s in strikes if s <= target_strike]
    upper = [s for s in strikes if s >= target_strike]

    if not lower:
        return vol_surface[upper[0]]
    if not upper:
        return vol_surface[lower[-1]]

    s1, s2 = lower[-1], upper[0]
    if s1 == s2:
        return vol_surface[s1]

    # Interpolation linéaire
    t = (target_strike - s1) / (s2 - s1)
    return vol_surface[s1] * (1 - t) + vol_surface[s2] * t


# --- KELLY CRITERION ---

def compute_kelly_size(fv, price, bankroll, kelly_fraction=0.5, min_size=1.0, max_size=10.0,
                       slippage_ticks=2, tick_size=0.01):
    """
    Taille optimale via Kelly Criterion pour une option binaire.

    Formule : f* = (fv - effective_price) / (1 - effective_price)
    Ajusté par kelly_fraction (0.5 = demi-Kelly, plus conservateur).
    Le slippage est soustrait de l'edge avant le calcul.

    Args:
        fv: fair value Black-Scholes
        price: prix d'achat (ask du marché)
        bankroll: capital disponible en USDC
        kelly_fraction: fraction de Kelly à utiliser (0.5 = demi)
        min_size: taille minimum en USDC
        max_size: taille maximum en USDC
        slippage_ticks: nombre de ticks de slippage estimé
        tick_size: taille d'un tick

    Returns:
        float taille en USDC, 0 si pas d'edge
    """
    # Ajuster le prix pour le slippage (on paiera plus cher que le ask affiché)
    effective_price = price + slippage_ticks * tick_size

    if fv <= effective_price or effective_price <= 0 or effective_price >= 1 or bankroll <= 0:
        return 0.0

    kelly_f = (fv - effective_price) / (1.0 - effective_price)
    size = kelly_fraction * kelly_f * bankroll
    return max(min_size, min(size, max_size))


# --- BLACK-SCHOLES BINAIRE ---

def calculate_black_scholes_binary(S, K, T_years, sigma):
    """
    S: Prix BTC
    K: Strike ($80,000)
    T_years: Temps en années
    """
    # Si le temps est écoulé ou négatif, la probabilité est binaire (0 ou 1)
    if T_years <= 0.00001: # On met une petite tolérance
        return 1.0 if S >= K else 0.0

    # Formule Black-Scholes Binaire
    d1 = (math.log(S / K) + (0.5 * sigma ** 2) * T_years) / (sigma * math.sqrt(T_years))
    d2 = d1 - sigma * math.sqrt(T_years)
    
    # Cumulative Distribution Function (CDF)
    nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    
    return nd2

def get_trading_signal(btc_price, poly_bid, poly_ask, strike, expiry_str, volatility, min_edge, side="NO"):

    # 1. CALCUL DU TEMPS RESTANT (Robuste)
    try:
        clean_expiry = expiry_str.replace("Z", "")
        expiry_dt = datetime.fromisoformat(clean_expiry)

        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = expiry_dt - now
        T = delta.total_seconds() / (365 * 24 * 3600)

    except Exception as e:
        print(f"⚠️ ERREUR DATE: {e} (Expiry: {expiry_str})")
        T = 0

    # Heures restantes pour l'affichage et la logique de sécurité
    hours_left = max(T * 365 * 24, 0)

    # 2. Calcul Fair Value (FV) du "YES"
    fv_yes = calculate_black_scholes_binary(btc_price, strike, T, volatility)

    # 3. Calcul de l'Edge
    signal = None
    edge = 0.0

    if side == "YES":
        if poly_ask > 0:
            edge = fv_yes - poly_ask
            if edge > min_edge:
                signal = "BUY_YES"
        return signal, edge, fv_yes, hours_left

    elif side == "NO":
        fv_no = 1.0 - fv_yes
        if poly_ask > 0:
            edge = fv_no - poly_ask
            if edge > min_edge:
                signal = "BUY_NO"
        return signal, edge, fv_no, hours_left

    return signal, edge, fv_yes, hours_left