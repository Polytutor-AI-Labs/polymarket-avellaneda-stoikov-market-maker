import os
from dotenv import load_dotenv

load_dotenv()

# --- 🔐 SÉCURITÉ ---
PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY")
if PRIVATE_KEY and PRIVATE_KEY.startswith("0x"): 
    PRIVATE_KEY = PRIVATE_KEY[2:]

# --- 🎯 CIBLE ---
# Le fichier généré par le scanneur
MARKET_DATA_FILE = "data/best_markets.json"
SIDE_TO_WATCH = "YES"  # "YES" ou "NO"

# --- 🧠 PARAMÈTRES MATHÉMATIQUES ---
# Fallback si Deribit est indisponible (la vol live remplace ça en temps réel)
VOLATILITY_FALLBACK = 0.40
STRIKE_PRICE = 0.0      # Sera rempli automatiquement par le JSON
EXPIRY_DATE_STR = ""    # Sera rempli automatiquement par le JSON

# --- ⚡ PARAMÈTRES DE TRADING ---
# Marge minimale pour tirer (Fair Value - Prix Marché)
# 0.03 = Il faut 3 cents d'écart pour qu'on tire (couvre le gaz + risque)
MIN_EDGE = 0.005 

# Budget max par tir (en USDC)
TRADE_SIZE_USDC = 5.0

# Sécurité : Ne jamais acheter au-dessus de ce prix (même si l'algo dit oui)
MAX_SAFE_PRICE = 0.95
MIN_SAFE_PRICE = 0.05          # Ne jamais acheter en dessous (YES ou NO)

# --- 🛡️ GESTION DU RISQUE ---
MAX_OPEN_POSITIONS = 5       # Nombre max de trades ouverts simultanément
MAX_EXPOSURE_USDC = 50.0     # Exposition max totale

# --- 📊 MAKER MODE ---
MAKER_OFFSET = 0.01       # Un tick au-dessus du best_bid (maker = 0% fees)
SELL_EDGE = 0.01           # Seuil pour auto-sell quand edge négatif
SELL_TIME_SAFETY_MINUTES = 30  # Vente de sécurité avant expiry

# Tick sizes Polymarket
TICK_SIZE_STANDARD = 0.01   # Prix entre 0.10 et 0.90
TICK_SIZE_EXTREME = 0.001   # Prix < 0.10 ou > 0.90

# --- 📐 MICROSTRUCTURE ---
MICROSTRUCTURE_ENABLED = True   # Ajuste l'edge via imbalance/momentum/spread

# --- 📈 SURFACE DE VOLATILITÉ ---
VOL_SURFACE_ENABLED = True      # Multi-strike Deribit (smile de vol)
VOL_SURFACE_STRIKES = 5         # Nombre de strikes autour d'ATM

# --- 🎰 KELLY CRITERION ---
KELLY_ENABLED = True            # Position sizing optimal
KELLY_FRACTION = 0.5            # Demi-Kelly (conservateur)
MIN_TRADE_USDC = 1.0            # Plancher minimal par trade

# --- 🎯 MULTI-STRIKE ---
MULTI_STRIKE_ENABLED = True     # Trader tous les marchés de best_markets.json

# --- 🔄 CANCEL & REPLACE ---
CANCEL_REPLACE_ENABLED = True   # Annuler les ordres stale et replacer
ORDER_TIMEOUT_SECONDS = 30      # Timeout avant cancel automatique

# --- 🏦 MARKET MAKER (AS-lite informed, spec v1.0) ---
MM_ENABLED = True               # True = MM continu, False = sniper legacy
MM_QUOTE_SIZE_USDC = 5.0        # USDC par quote (min 5 shares imposé par Polymarket)

# Spread & skew (AS-lite formula: center = FV - (net_inv/max_inv) * SKEW_MAX)
MM_HALF_SPREAD_CENTS = 3        # Demi-spread par côté en cents (défaut spec: 3)
MM_SKEW_MAX = 0.02              # Skew max d'inventaire en $ (2 cents à inventaire plein)

# Limites d'inventaire
MM_MAX_INVENTORY = 10           # Net shares max par marché — serré pour limiter l'exposition sur jours tendance
MM_MAX_GROSS_SHARES = 20        # Brut max par marché (YES + NO)
MM_CROSS_STRIKE_MULT = 1.2      # Multiplicateur cross-strike exposure (brut)
MM_CROSS_DIRECTIONAL_MULT = 0.8 # Multiplicateur cross-strike directionnel

# Gestion des ordres
MM_REQUOTE_INTERVAL = 5.0       # Secondes entre refresh forcé des quotes
MM_ORDER_TOLERANCE_TICKS = 2    # Ticks de drift avant cancel+replace (spec: 2 ticks)
MM_POLL_FILLS_INTERVAL = 10.0   # Secondes entre polling actif des fills

# GTD & phases temporelles
MM_ORDER_EXPIRATION_BUFFER = 90 # Secondes avant expiry pour tuer les ordres
MM_WIND_DOWN_SECS = 120         # Secondes restantes → phase WIND_DOWN (cancel bids)
MM_STOP_SECS = 30               # Secondes restantes → phase STOP (cancel all + FAK)

# Hybrid maker/taker
MM_TAKER_EDGE_THRESHOLD = 0.15  # Edge > 15% → FAK taker (vitesse > rebate maker)

# Filtre momentum anti-adverse-selection
MM_MOMENTUM_FILTER = True       # Activer le filtre momentum
MM_MOMENTUM_THRESHOLD = 0.003   # 0.3% de move BTC → pull les bids du côté perdant

# Circuit breakers
MM_MAX_DAILY_LOSS = -8.0        # Perte max quotidienne (réalisé + unrealized)
MM_VOL_SPIKE_THRESHOLD = 0.02   # Move BTC % en 10s qui déclenche halt
MM_VOL_SPIKE_WINDOW = 10        # Fenêtre en secondes pour spike detection
MM_MIN_HOURS_TO_EXPIRY = 0.5    # Fallback stop quoting (= MM_WIND_DOWN_SECS / 3600)
MM_MIN_SENSITIVITY = 0.04       # Sensibilité binaire min fv*(1-fv) pour quoter

# --- 📊 DELTA HEDGE (Phase 2 : Hyperliquid) ---
HEDGE_ENABLED = True                # Activer le calcul de delta
HEDGE_DRY_RUN = True               # auto-patched by check_pnl.py
HEDGE_REBALANCE_INTERVAL = 30.0     # Secondes entre chaque recalcul de delta
HEDGE_DELTA_TOLERANCE = 0.002       # BTC de drift avant rehedge (était 0.005 → trop haut, hedge ne déclenchait jamais)
HEDGE_PLATFORM = "hyperliquid"      # Plateforme de hedge
HEDGE_MAX_POSITION_BTC = 0.05       # Position max BTC sur la plateforme de hedge
HEDGE_MAX_SINGLE_TRADE_BTC = 0.005  # Taille max par rebalance (était 0.01, granularité trop grosse)
HEDGE_LEVERAGE = 10                 # 10x cross pour réduire la marge requise ($335/trade < $300 à 1x)
HEDGE_SLIPPAGE = 0.01               # 1% slippage max market orders

# --- PAPER TRADING (SIM fills) ---
SIM_FILL_MODEL = "stochastic"       # "deterministic" ou "stochastic"
SIM_FILL_CHECK_INTERVAL = 1.0       # Secondes entre checks (rate limiter)
SIM_FILL_LOG_TO_CSV = True          # Logger les fills SIM dans trades_journal.csv

# --- 🛠️ MODE ---
LIVE_MODE = False   # auto-patched by check_pnl.py