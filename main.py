import connectors.cf_bypass  # AVANT tout import de py_clob_client
import asyncio
import json
import os
import csv
import time
import requests
from datetime import datetime, timezone as _tz
import config as cfg
from connectors.binance_ws import BinanceConnector
from connectors.poly_ws import PolyConnector
from connectors.deribit_ws import DeribitConnector
from strategy.pricing import get_trading_signal, interpolate_iv, compute_kelly_size
from strategy.microstructure import MicrostructureAnalyzer
from execution.trader import Sniper, create_clob_client, compute_maker_buy_price, _tick_size
from data.recorder import DataRecorder
from execution.hedge_manager import HedgeManager


# --- CONTEXTE PAR MARCHÉ ---

class MarketContext:
    """État isolé pour chaque marché/strike."""
    def __init__(self, market_data):
        self.question = market_data["question"]
        self.strike = market_data["strike"]
        self.expiry = market_data["expiry"]
        self.yes_token_id = market_data["yes_id"]
        self.no_token_id = market_data["no_id"]
        # YES orderbook
        self.yes_bid = 0.0
        self.yes_ask = 0.0
        self.yes_bid_vol = 0.0
        self.yes_ask_vol = 0.0
        self.yes_bid_depth = []
        self.yes_ask_depth = []
        # NO orderbook
        self.no_bid = 0.0
        self.no_ask = 0.0
        self.no_bid_vol = 0.0
        self.no_ask_vol = 0.0
        # Legacy compat (sniper mode uses these)
        self.poly_bid = 0.0
        self.poly_ask = 0.0
        self.poly_bid_vol = 0.0
        self.poly_ask_vol = 0.0
        self.bid_depth = []
        self.ask_depth = []
        # Positions (legacy)
        self.positions = {}
        # Cancel & Replace (legacy)
        self.pending_order = None
        # Inventory MM
        self.yes_shares = 0.0
        self.no_shares = 0.0
        # VWAP cost basis (total USDC dépensé pour les shares en inventaire)
        self.yes_cost_basis = 0.0
        self.no_cost_basis = 0.0
        # Tape des trades (pour RL data collection)
        self.last_trade_price = 0.0
        self.last_trade_size = 0.0
        self.last_trade_side = ""  # "BUY" ou "SELL"
        # VPIN calculator (toxicité du flow)
        from strategy.microstructure import VPINCalculator
        self.vpin_calc = VPINCalculator(window_trades=50, toxicity_threshold=0.35, penalty_factor=1.5)

    def update_yes_book(self, bids, asks):
        if asks:
            self.yes_ask = float(asks[0]['price'])
            self.yes_ask_vol = float(asks[0]['size'])
            self.yes_ask_depth = [(float(a['price']), float(a['size'])) for a in asks[1:5]]  # niveaux 2-5
        if bids:
            self.yes_bid = float(bids[0]['price'])
            self.yes_bid_vol = float(bids[0]['size'])
            self.yes_bid_depth = [(float(b['price']), float(b['size'])) for b in bids[1:5]]  # niveaux 2-5
        # Legacy compat
        self.poly_bid = self.yes_bid
        self.poly_ask = self.yes_ask
        self.poly_bid_vol = self.yes_bid_vol
        self.poly_ask_vol = self.yes_ask_vol
        self.bid_depth = self.yes_bid_depth
        self.ask_depth = self.yes_ask_depth

    def update_no_book(self, bids, asks):
        if asks:
            self.no_ask = float(asks[0]['price'])
            self.no_ask_vol = float(asks[0]['size'])
        if bids:
            self.no_bid = float(bids[0]['price'])
            self.no_bid_vol = float(bids[0]['size'])

    # Legacy methods for sniper mode
    def update_poly_book(self, bids, asks):
        self.update_yes_book(bids, asks)

    def add_position(self, token_id, shares):
        self.positions[token_id] = self.positions.get(token_id, 0) + shares

    def remove_position(self, token_id, shares):
        current = self.positions.get(token_id, 0)
        remaining = current - shares
        if remaining <= 0.1:
            self.positions.pop(token_id, None)
        else:
            self.positions[token_id] = remaining

    def get_position_size(self):
        token_id = self.yes_token_id if cfg.SIDE_TO_WATCH == "YES" else self.no_token_id
        return self.positions.get(token_id, 0)

    @property
    def net_inventory(self):
        """q = yes_shares - no_shares pour Avellaneda-Stoikov."""
        return self.yes_shares - self.no_shares

    @property
    def strike_label(self):
        return f"${self.strike/1000:.0f}k"

    @property
    def unrealized_pnl(self):
        """P&L non-réalisé basé sur les bids actuels du marché."""
        pnl = 0.0
        if self.yes_shares > 0 and self.yes_bid > 0:
            yes_vwap = self.yes_cost_basis / self.yes_shares if self.yes_shares > 0 else 0
            pnl += self.yes_shares * (self.yes_bid - yes_vwap)
        if self.no_shares > 0 and self.no_bid > 0:
            no_vwap = self.no_cost_basis / self.no_shares if self.no_shares > 0 else 0
            pnl += self.no_shares * (self.no_bid - no_vwap)
        return pnl

    def load_positions_from_journal(self, trades_file):
        """Charge les positions et cost_basis depuis le journal en rejouant les fills."""
        if not os.path.exists(trades_file):
            return
        try:
            with open(trades_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tid = row.get("token_id", "")
                    side = row.get("side", "")
                    shares = float(row.get("size_shares", 0))
                    price = float(row.get("price", 0))
                    status = row.get("pnl_status", "")

                    # Seuls les FILLs comptent — MM_LIVE = ordre posé, pas une position
                    buy_statuses = ("OPEN", "MM_FILL_BUY", "SIM_FILL_BUY")
                    sell_statuses = ("CLOSED", "MM_FILL_SELL", "MM_PANIC_CLOSE", "SIM_FILL_SELL")
                    is_sync = (row.get("order_id", "") == "SYNC_ONCHAIN")

                    if tid == self.yes_token_id:
                        if side == "BUY" and (status in buy_statuses or is_sync):
                            self.yes_shares += shares
                            self.yes_cost_basis += price * shares
                            self.add_position(tid, shares)
                        elif side == "SELL" and (status in sell_statuses or is_sync):
                            # Retirer le cost_basis proportionnellement (VWAP)
                            if self.yes_shares > 0:
                                avg = self.yes_cost_basis / self.yes_shares
                                self.yes_cost_basis -= avg * min(shares, self.yes_shares)
                            self.yes_shares = max(0, self.yes_shares - shares)
                            if self.yes_shares <= 0.01:
                                self.yes_cost_basis = 0.0
                                self.yes_shares = 0.0
                            self.remove_position(tid, shares)
                    elif tid == self.no_token_id:
                        if side == "BUY" and (status in buy_statuses or is_sync):
                            self.no_shares += shares
                            self.no_cost_basis += price * shares
                        elif side == "SELL" and (status in sell_statuses or is_sync):
                            if self.no_shares > 0:
                                avg = self.no_cost_basis / self.no_shares
                                self.no_cost_basis -= avg * min(shares, self.no_shares)
                            self.no_shares = max(0, self.no_shares - shares)
                            if self.no_shares <= 0.01:
                                self.no_cost_basis = 0.0
                                self.no_shares = 0.0

            if self.yes_shares > 0 or self.no_shares > 0:
                yes_vwap = self.yes_cost_basis / self.yes_shares if self.yes_shares > 0 else 0
                no_vwap = self.no_cost_basis / self.no_shares if self.no_shares > 0 else 0
                print(f"   pos YES:{self.yes_shares:.1f}@{yes_vwap:.3f} NO:{self.no_shares:.1f}@{no_vwap:.3f} net:{self.net_inventory:.1f}")
        except Exception as e:
            print(f"   !! Erreur chargement positions : {e}")


# --- STATE MANAGER GLOBAL ---

class BotState:
    """État partagé : prix BTC, volatilité."""
    def __init__(self):
        self.btc_price = 0.0
        self.implied_vol = 0.0
        self.vol_surface = {}
        self.expiry = ""
        self.positions = {}
        self.funding_rate = 0.0    # BTC perp funding rate Binance (mise à jour ~60s)

    def update_btc(self, price):
        self.btc_price = price


# --- RECONCILIATION ON-CHAIN ---

WALLET_ADDRESS = "0x69D254a20982fec305146830D3984aAD733149bA"

def reconcile_positions(markets, recorder):
    """Vérifie les positions on-chain via data-api et corrige les drifts."""
    try:
        url = f"https://data-api.polymarket.com/positions?user={WALLET_ADDRESS}"
        r = requests.get(url, timeout=10)
        api_positions = {p["asset"]: float(p.get("size", 0)) for p in r.json() if float(p.get("size", 0)) > 0.1}
    except Exception as e:
        print(f"   !! Reconciliation échouée: {e}")
        return

    corrections = 0
    for ctx in markets:
        for token_id, attr, cost_attr in [
            (ctx.yes_token_id, "yes_shares", "yes_cost_basis"),
            (ctx.no_token_id, "no_shares", "no_cost_basis"),
        ]:
            api_size = api_positions.get(token_id, 0)
            local_size = getattr(ctx, attr)
            drift = api_size - local_size

            if abs(drift) < 0.5:
                continue

            corrections += 1
            side_label = "YES" if "yes" in attr else "NO"
            # Estimer le prix moyen pour le drift
            if "yes" in attr:
                est_price = ctx.yes_bid if ctx.yes_bid > 0 else 0.5
            else:
                est_price = ctx.no_bid if ctx.no_bid > 0 else 0.5

            print(f"   !! SYNC {ctx.strike_label} {side_label}: journal={local_size:.1f} chain={api_size:.1f} drift={drift:+.1f}")
            setattr(ctx, attr, api_size)

            if drift > 0:
                # Shares manquantes — ajouter au cost_basis
                setattr(ctx, cost_attr, getattr(ctx, cost_attr) + drift * est_price)
                recorder.log_execution(
                    order_id="SYNC_ONCHAIN",
                    side="BUY", token_id=token_id,
                    price=est_price, size_shares=drift,
                    status="MM_FILL_BUY"
                )
            else:
                # Shares en trop dans le journal — réduire
                avg = getattr(ctx, cost_attr) / local_size if local_size > 0 else est_price
                setattr(ctx, cost_attr, max(0, getattr(ctx, cost_attr) + drift * avg))
                recorder.log_execution(
                    order_id="SYNC_ONCHAIN",
                    side="SELL", token_id=token_id,
                    price=est_price, size_shares=abs(drift),
                    status="MM_FILL_SELL"
                )

    if corrections:
        print(f"   >> Reconciliation: {corrections} correction(s) appliquée(s)")


# --- POLY ADAPTERS ---

class PolyAdapterYES(PolyConnector):
    """WS Polymarket pour le token YES d'un MarketContext."""
    def _process(self, data):
        if data.get("event_type") == "book":
            bids = sorted(data.get("bids", []), key=lambda x: float(x.get("price", 0)), reverse=True)
            asks = sorted(data.get("asks", []), key=lambda x: float(x.get("price", 999)))
            self.state.update_yes_book(bids, asks)
        elif data.get("event_type") == "price_change":
            # Capture du dernier trade pour la tape (RL data collection)
            changes = data.get("price_changes") or data.get("changes") or []
            for c in changes:
                s = float(c.get("size", 0))
                if s > 0:
                    self.state.last_trade_price = float(c.get("price", 0))
                    self.state.last_trade_size = s
                    self.state.last_trade_side = c.get("side", "")
                    self.state.vpin_calc.add_trade(
                        price=self.state.last_trade_price,
                        size=self.state.last_trade_size,
                        side=self.state.last_trade_side,
                    )


class PolyAdapterNO(PolyConnector):
    """WS Polymarket pour le token NO d'un MarketContext."""
    def _process(self, data):
        if data.get("event_type") == "book":
            bids = sorted(data.get("bids", []), key=lambda x: float(x.get("price", 0)), reverse=True)
            asks = sorted(data.get("asks", []), key=lambda x: float(x.get("price", 999)))
            self.state.update_no_book(bids, asks)
        elif data.get("event_type") == "price_change":
            # Capture du dernier trade pour la tape (RL data collection)
            changes = data.get("price_changes") or data.get("changes") or []
            for c in changes:
                s = float(c.get("size", 0))
                if s > 0:
                    self.state.last_trade_price = float(c.get("price", 0))
                    self.state.last_trade_size = s
                    self.state.last_trade_side = c.get("side", "")
                    self.state.vpin_calc.add_trade(
                        price=self.state.last_trade_price,
                        size=self.state.last_trade_size,
                        side=self.state.last_trade_side,
                    )


# --- BOUCLE MARKET MAKER ---

async def run_market_maker(markets, state, recorder, micro):
    """Boucle principale mode Market Maker Avellaneda-Stoikov."""
    from strategy.avellaneda_stoikov import AvellanedaStoikov
    from execution.quote_manager import QuoteManager
    from strategy.risk_manager import RiskManager
    from strategy.realized_vol import RealizedVolCalculator, SigmaKalmanFilter

    # Moteur AS-lite (spec v1.0 — centré sur BS FV, pas sur market mid)
    as_engine = AvellanedaStoikov(
        half_spread=cfg.MM_HALF_SPREAD_CENTS * 0.01,
        skew_max=cfg.MM_SKEW_MAX,
        max_inventory=cfg.MM_MAX_INVENTORY,
        max_gross=cfg.MM_MAX_GROSS_SHARES,
        size_per_quote=cfg.MM_QUOTE_SIZE_USDC,
    )

    # Quote Manager
    if cfg.LIVE_MODE:
        client = create_clob_client(cfg.PRIVATE_KEY)
        print("MM: Client CLOB authentifié")
    else:
        client = None

    quote_mgr = QuoteManager(
        client=client,
        recorder=recorder,
        requote_interval=cfg.MM_REQUOTE_INTERVAL,
        price_drift_ticks=cfg.MM_ORDER_TOLERANCE_TICKS,
        order_timeout=90.0,   # > GTD lifetime (120s) sauf buffer 30s → cancel juste avant expiry naturelle
        live_mode=cfg.LIVE_MODE,
    )

    # --- CANCEL-ALL AU DÉMARRAGE (sécurité crash — pilier 2 de la spec) ---
    if cfg.LIVE_MODE:
        quote_mgr.cancel_everything()
        print("MM: Cancel-all au démarrage (ordres stale potentiels effacés)")

    # Risk Manager
    risk_mgr = RiskManager(
        max_daily_loss=cfg.MM_MAX_DAILY_LOSS,
        vol_spike_threshold=cfg.MM_VOL_SPIKE_THRESHOLD,
        vol_spike_window=cfg.MM_VOL_SPIKE_WINDOW,
        min_hours_to_expiry=cfg.MM_MIN_HOURS_TO_EXPIRY,
    )

    # Realized Vol Calculator (OPTIMISATION 2)
    rvol_calc = RealizedVolCalculator(
        window_minutes=60,
        blend_weight_realized=0.3,  # conservé pour compatibilité stats
    )

    # Filtre de Kalman pour fusion optimale Deribit IV + realized vol
    sigma_kalman = SigmaKalmanFilter(initial_sigma=cfg.VOLATILITY_FALLBACK)

    # Delta Hedge Manager (Phase 2 : Hyperliquid)
    # IMPORTANT: HyperliquidClient.__init__ fait des appels HTTP synchrones (update_leverage).
    # On le crée dans un thread pour ne pas bloquer l'event loop asyncio.
    hedge_mgr = None
    hl_client = None
    if cfg.HEDGE_ENABLED:
        if not cfg.HEDGE_DRY_RUN:
            def _create_hl_client():
                from connectors.hyperliquid_client import HyperliquidClient
                return HyperliquidClient(
                    dry_run=False,
                    max_position_btc=cfg.HEDGE_MAX_POSITION_BTC,
                    max_single_trade_btc=cfg.HEDGE_MAX_SINGLE_TRADE_BTC,
                    leverage=cfg.HEDGE_LEVERAGE,
                    slippage=cfg.HEDGE_SLIPPAGE,
                )
            try:
                loop = asyncio.get_event_loop()
                hl_client = await asyncio.wait_for(
                    loop.run_in_executor(None, _create_hl_client),
                    timeout=10.0,
                )
                print(f"MM: Hyperliquid client OK (wallet={hl_client.address[:10]}...)")
            except (asyncio.TimeoutError, TimeoutError):
                print("!! Hyperliquid init TIMEOUT (10s) — fallback dry-run")
                hl_client = None
            except BaseException as e:
                print(f"!! Hyperliquid init FAILED: {type(e).__name__}: {e} — fallback dry-run")
                hl_client = None

        hedge_mgr = HedgeManager(
            rebalance_interval=cfg.HEDGE_REBALANCE_INTERVAL,
            delta_tolerance=cfg.HEDGE_DELTA_TOLERANCE,
            dry_run=cfg.HEDGE_DRY_RUN or hl_client is None,
            hl_client=hl_client,
        )
        mode_str = "LIVE Hyperliquid" if hl_client else "dry-run"
        print(f"MM: Delta hedge actif ({mode_str}, interval={cfg.HEDGE_REBALANCE_INTERVAL}s)")

    # --- PHASES TEMPORELLES (QUOTING / WIND_DOWN / STOP) ---
    # Tracker par marché — transitions déclenchées par timer dédié 1s
    market_phases = {f"{ctx.strike}": "QUOTING" for ctx in markets}
    last_phase_check = 0.0

    def _compute_gtd_expiry(expiry_str):
        """GTD timestamp: min(120s, time_left - BUFFER) clampé à 60s min."""
        try:
            clean = expiry_str.replace("Z", "")
            expiry_dt = datetime.fromisoformat(clean)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=_tz.utc)
            time_left_s = (expiry_dt - datetime.now(_tz.utc)).total_seconds()
            desired = min(120.0, time_left_s - cfg.MM_ORDER_EXPIRATION_BUFFER)
            lifetime = max(60.0, desired)
            return int(time.time() + lifetime)
        except Exception:
            return int(time.time() + 60)

    # P&L tracking
    realized_pnl = 0.0
    total_fills = 0

    tick_count = 0
    last_poll_time = {}   # {market_key: timestamp} — poll indépendant par marché
    last_reconcile_time = 0
    last_panic_close_time = 0
    last_hedge_log_time = 0
    RECONCILE_INTERVAL = 300  # 5 minutes
    # Limite d'exposition cross-strike (utilise config)
    CROSS_STRIKE_MAX_EXPOSURE = cfg.MM_MAX_INVENTORY * cfg.MM_CROSS_STRIKE_MULT
    CROSS_STRIKE_MAX_DIRECTIONAL = cfg.MM_MAX_INVENTORY * cfg.MM_CROSS_DIRECTIONAL_MULT
    print(f"MM: Boucle active — {len(markets)} marché(s)")

    try:
        while True:
            if state.btc_price <= 0:
                await asyncio.sleep(0.1)
                continue

            # Update risk manager avec BTC
            risk_mgr.update_btc(state.btc_price)
            if micro:
                micro.update_btc(state.btc_price)

            # OPTIMISATION 2: Feed realized vol calculator
            rvol_calc.add_price(state.btc_price)

            # --- PHASE TIMER (résolution 1s, indépendant du quote refresh) ---
            _now = time.time()
            if _now - last_phase_check >= 1.0:
                for _ctx in markets:
                    _mkey = f"{_ctx.strike}"
                    _prev = market_phases[_mkey]
                    # Calculer secs restantes depuis expiry
                    try:
                        _clean = _ctx.expiry.replace("Z", "")
                        _exp_dt = datetime.fromisoformat(_clean)
                        if _exp_dt.tzinfo is None:
                            _exp_dt = _exp_dt.replace(tzinfo=_tz.utc)
                        _secs_left = (_exp_dt - datetime.now(_tz.utc)).total_seconds()
                    except Exception:
                        _secs_left = 9999

                    if _secs_left <= cfg.MM_STOP_SECS and _prev != "STOP":
                        market_phases[_mkey] = "STOP"
                        print(f"!! PHASE STOP {_ctx.strike_label}: cancel all ordres")
                        quote_mgr.cancel_all_market(_mkey)
                        # FAK fire-sale si inventaire résiduel
                        _t = _tick_size(_ctx.yes_bid if _ctx.yes_bid > 0 else 0.5)
                        if _ctx.net_inventory > 5 and _ctx.yes_shares >= 5:
                            _px = max(_t, (_ctx.yes_bid or _t) - 2 * _t)
                            quote_mgr.post_market_order(_ctx.yes_token_id, "SELL", _px, _ctx.yes_shares)
                            print(f"   -> FAK SELL {_ctx.yes_shares:.0f} YES @ {_px:.3f}")
                        elif _ctx.net_inventory < -5 and _ctx.no_shares >= 5:
                            _px = max(_t, (_ctx.no_bid or _t) - 2 * _t)
                            quote_mgr.post_market_order(_ctx.no_token_id, "SELL", _px, _ctx.no_shares)
                            print(f"   -> FAK SELL {_ctx.no_shares:.0f} NO @ {_px:.3f}")

                    elif _secs_left <= cfg.MM_WIND_DOWN_SECS and _prev == "QUOTING":
                        market_phases[_mkey] = "WIND_DOWN"
                        print(f"MM: PHASE WIND_DOWN {_ctx.strike_label} ({_secs_left:.0f}s restantes)")
                        quote_mgr.cancel_side(_mkey, "BUY")
                last_phase_check = _now

            vol = state.implied_vol if state.implied_vol > 0 else cfg.VOLATILITY_FALLBACK
            for ctx in markets:
                if ctx.yes_ask <= 0:
                    continue

                market_key = f"{ctx.strike}"

                # --- VOLATILITÉ ---
                # Récupérer IV Deribit
                if cfg.VOL_SURFACE_ENABLED and state.vol_surface:
                    deribit_iv = interpolate_iv(state.vol_surface, ctx.strike)
                    if deribit_iv is None:
                        deribit_iv = state.implied_vol if state.implied_vol > 0 else cfg.VOLATILITY_FALLBACK
                    vol_src = "S"
                elif state.implied_vol > 0:
                    deribit_iv = state.implied_vol
                    vol_src = ""
                else:
                    deribit_iv = cfg.VOLATILITY_FALLBACK
                    vol_src = "*"

                # Filtre de Kalman : fusion optimale Deribit IV + realized vol
                # sigma_kalman est partagé entre les marchés (même asset BTC)
                realized = rvol_calc.get_realized_vol()
                vol = sigma_kalman.update(deribit_iv, realized)
                vol_src += "K"  # Marker Kalman actif

                # --- FAIR VALUE (Black-Scholes) ---
                _, _, fv_yes, hours_left = get_trading_signal(
                    btc_price=state.btc_price,
                    poly_bid=ctx.yes_bid,
                    poly_ask=ctx.yes_ask,
                    strike=ctx.strike,
                    expiry_str=ctx.expiry,
                    volatility=vol,
                    min_edge=cfg.MIN_EDGE,
                    side="YES"
                )

                # --- SENSIBILITÉ BINAIRE CHECK ---
                # Marchés deep ITM/OTM ont une sensibilité ~0 → pas rentable de quoter
                _sensitivity = fv_yes * (1.0 - fv_yes)
                if _sensitivity < cfg.MM_MIN_SENSITIVITY:
                    if tick_count % 500 == 0:
                        print(f"   SKIP {ctx.strike_label}: FV={fv_yes:.4f} sensitivity={_sensitivity:.4f} < {cfg.MM_MIN_SENSITIVITY}")
                    quote_mgr.cancel_all_market(market_key)
                    continue

                # --- RISK CHECK ---
                inventory_abs = abs(ctx.net_inventory)
                risk = risk_mgr.check(hours_left, inventory_abs)
                if risk["halt"]:
                    quote_mgr.cancel_all_market(market_key)
                    # Cancel immédiat global si demandé (daily loss)
                    if risk.get("cancel_immediate"):
                        quote_mgr.cancel_everything()
                    if tick_count % 100 == 0:
                        print(f"!! HALT {ctx.strike_label}: {risk['reason']}")
                    # Vente d'urgence si halt ET panic_close (daily loss avec inventaire)
                    if risk.get("panic_close"):
                        now_pc = time.time()
                        if now_pc - last_panic_close_time > 30:
                            tick_pc = _tick_size(ctx.yes_bid if ctx.yes_bid > 0 else 0.5)
                            if ctx.net_inventory > 5 and ctx.yes_shares >= 5:
                                sell_sz = min(ctx.net_inventory, ctx.yes_shares)
                                sell_px = max(tick_pc, (ctx.yes_bid or tick_pc) - 2 * tick_pc)
                                if sell_px > 0.01:
                                    quote_mgr.post_market_order(ctx.yes_token_id, "SELL", sell_px, sell_sz)
                                    print(f"   -> HALT SELL {sell_sz:.1f} YES @ {sell_px:.3f} (stop-loss)")
                            elif ctx.net_inventory < -5 and ctx.no_shares >= 5:
                                sell_sz = min(abs(ctx.net_inventory), ctx.no_shares)
                                sell_px = max(tick_pc, (ctx.no_bid or tick_pc) - 2 * tick_pc)
                                if sell_px > 0.01:
                                    quote_mgr.post_market_order(ctx.no_token_id, "SELL", sell_px, sell_sz)
                                    print(f"   -> HALT SELL {sell_sz:.1f} NO @ {sell_px:.3f} (stop-loss)")
                            last_panic_close_time = now_pc
                    continue

                # OPTIMISATION 3: PANIC CLOSE (< 2h avant expiry et inventory > 20)
                if risk.get("panic_close", False):
                    now = time.time()
                    if now - last_panic_close_time > 30:  # Toutes les 30s (accéléré)
                        print(f"PANIC CLOSE {ctx.strike_label}: {hours_left:.1f}h left, inventory={ctx.net_inventory:.0f}")
                        # Cancel tous les ordres d'achat d'abord
                        quote_mgr.cancel_side(market_key, "BUY")
                        # Prix agressif : bid - 2 ticks pour forcer l'exécution
                        tick = _tick_size(ctx.yes_bid if ctx.yes_bid > 0 else 0.5)
                        if ctx.net_inventory > 5:
                            sell_size = min(ctx.net_inventory, ctx.yes_shares)
                            aggressive_price = max(tick, ctx.yes_bid - 2 * tick)
                            if sell_size >= 5 and aggressive_price > 0.01:
                                quote_mgr.post_market_order(ctx.yes_token_id, "SELL", aggressive_price, sell_size)
                                print(f"   -> SELL {sell_size:.1f} YES @ {aggressive_price:.3f} (aggressive)")
                        elif ctx.net_inventory < -5:
                            sell_size = min(abs(ctx.net_inventory), ctx.no_shares)
                            aggressive_price = max(tick, ctx.no_bid - 2 * tick)
                            if sell_size >= 5 and aggressive_price > 0.01:
                                quote_mgr.post_market_order(ctx.no_token_id, "SELL", aggressive_price, sell_size)
                                print(f"   -> SELL {sell_size:.1f} NO @ {aggressive_price:.3f} (aggressive)")
                        last_panic_close_time = now

                # --- CROSS-STRIKE EXPOSURE CHECK ---
                # Calcule l'exposition directionnelle totale : somme de yes_shares et
                # no_shares sur TOUS les marches. Bloque les achats si trop expose.
                total_yes_exposure = sum(m.yes_shares for m in markets)
                total_no_exposure = sum(m.no_shares for m in markets)
                cross_exposure = total_yes_exposure + total_no_exposure
                cross_directional = abs(total_yes_exposure - total_no_exposure)

                cross_limit_breached = (
                    cross_exposure > CROSS_STRIKE_MAX_EXPOSURE
                    or cross_directional > CROSS_STRIKE_MAX_DIRECTIONAL
                )
                if cross_limit_breached:
                    now_hedge = time.time()
                    if now_hedge - last_hedge_log_time > 60:
                        print(
                            f"!! HEDGE BLOCK: brut={cross_exposure:.0f}/{CROSS_STRIKE_MAX_EXPOSURE:.0f} "
                            f"dir={cross_directional:.0f}/{CROSS_STRIKE_MAX_DIRECTIONAL:.0f} "
                            f"(YES={total_yes_exposure:.0f} NO={total_no_exposure:.0f})"
                        )
                        last_hedge_log_time = now_hedge
                    # Cancel immédiat des BUY existants pour stopper l'hémorragie
                    quote_mgr.cancel_side(market_key, "BUY")

                # HARD INVENTORY LIMITS : 80% du max_inventory config
                hard_limit_inventory = int(cfg.MM_MAX_INVENTORY * 0.8)
                if inventory_abs > hard_limit_inventory:
                    if tick_count % 100 == 0:
                        print(f"⚠️ HARD LIMIT {ctx.strike_label}: inventory={ctx.net_inventory:.0f} > {hard_limit_inventory}")
                    # Pull le côté qui aggraverait l'inventory
                    if ctx.net_inventory > hard_limit_inventory:
                        # Trop long YES → cancel BIDs (stop d'achat)
                        quote_mgr.cancel_side(market_key, "BUY")
                    elif ctx.net_inventory < -hard_limit_inventory:
                        # Trop short YES (trop long NO) → cancel BIDs uniquement
                        # NE PAS cancel les SELLs : les NO_ASK permettent de sortir la position
                        quote_mgr.cancel_side(market_key, "BUY")

                # --- MICROSTRUCTURE (ajustement optionnel) ---
                micro_adj = 0.0
                micro_momentum = 0.0
                micro_spread_score = 0.0
                if micro:
                    spread = ctx.yes_ask - ctx.yes_bid
                    ms = micro.compute_signals(
                        ctx.yes_bid_vol, ctx.yes_ask_vol,
                        spread, state.btc_price, ctx.strike, "YES"
                    )
                    micro_adj = ms.get("edge_adjustment", 0)
                    micro_momentum = ms.get("momentum", 0)
                    micro_spread_score = ms.get("spread_score", 0)

                # --- TICK SIZE ---
                tick = _tick_size(fv_yes)

                # --- IMBALANCE pour spread dynamique ---
                _total_vol = ctx.yes_bid_vol + ctx.yes_ask_vol
                _imbalance = (ctx.yes_bid_vol - ctx.yes_ask_vol) / _total_vol if _total_vol > 0 else 0.0

                # --- FILTRE MOMENTUM (anti-adverse-selection) ---
                # Si BTC monte fort → ne pas acheter NO (on serait contre la tendance)
                # Si BTC descend fort → ne pas acheter YES (idem)
                # Évite d'accumuler l'inventaire du mauvais côté sur les jours de tendance
                _momentum_block_yes_bid = False
                _momentum_block_no_bid  = False
                if cfg.MM_MOMENTUM_FILTER and micro_momentum != 0:
                    _thr = cfg.MM_MOMENTUM_THRESHOLD
                    if micro_momentum < -_thr:
                        # BTC descend → ne pas acheter YES (YES va baisser)
                        _momentum_block_yes_bid = True
                    elif micro_momentum > _thr:
                        # BTC monte → ne pas acheter NO (NO va baisser)
                        _momentum_block_no_bid = True

                # --- HEDGE HEALTH → spread penalty ---
                _spread_mult = risk["spread_multiplier"]
                _hedge_critical = False
                if hedge_mgr:
                    _hedge_health = hedge_mgr.get_health()
                    _spread_mult *= _hedge_health["spread_penalty"]
                    _hedge_critical = _hedge_health["critical"]
                # VPIN spread penalty (flow toxicity)
                _vpin_penalty = ctx.vpin_calc.get_spread_penalty()
                _spread_mult *= _vpin_penalty

                # --- PHASE CHECK : skip si STOP, pull bids si WIND_DOWN ---
                _phase = market_phases.get(market_key, "QUOTING")
                if _phase == "STOP":
                    continue

                # --- HYBRID MAKER/TAKER : FAK si edge > seuil (15%) ---
                if cfg.MM_TAKER_EDGE_THRESHOLD > 0:
                    _yes_edge = fv_yes - ctx.yes_ask if ctx.yes_ask > 0 else 0
                    _no_fv    = 1.0 - fv_yes
                    _no_edge  = _no_fv - ctx.no_ask if ctx.no_ask > 0 else 0
                    if _yes_edge > cfg.MM_TAKER_EDGE_THRESHOLD and ctx.yes_shares < cfg.MM_MAX_INVENTORY:
                        _taker_sz = cfg.MM_QUOTE_SIZE_USDC / ctx.yes_ask if ctx.yes_ask > 0 else 0
                        if cfg.LIVE_MODE and _taker_sz >= 5:
                            from py_clob_client.clob_types import OrderArgs as _OA, OrderType as _OT
                            _args = _OA(price=ctx.yes_ask, size=_taker_sz,
                                        side="BUY", token_id=ctx.yes_token_id)
                            try:
                                _r = client.create_and_post_order(_args, order_type=_OT.FOK)
                                if _r and _r.get("success"):
                                    print(f"   >> FAK TAKER YES {_taker_sz:.1f}@{ctx.yes_ask:.3f} edge={_yes_edge:+.3f}")
                            except Exception:
                                pass
                    elif _no_edge > cfg.MM_TAKER_EDGE_THRESHOLD and ctx.no_shares < cfg.MM_MAX_INVENTORY:
                        _taker_sz = cfg.MM_QUOTE_SIZE_USDC / ctx.no_ask if ctx.no_ask > 0 else 0
                        if cfg.LIVE_MODE and _taker_sz >= 5:
                            from py_clob_client.clob_types import OrderArgs as _OA, OrderType as _OT
                            _args = _OA(price=ctx.no_ask, size=_taker_sz,
                                        side="BUY", token_id=ctx.no_token_id)
                            try:
                                _r = client.create_and_post_order(_args, order_type=_OT.FOK)
                                if _r and _r.get("success"):
                                    print(f"   >> FAK TAKER NO {_taker_sz:.1f}@{ctx.no_ask:.3f} edge={_no_edge:+.3f}")
                            except Exception:
                                pass

                # --- AS-LITE QUOTES (centré sur BS FV, pas sur market mid) ---
                quote_set = as_engine.compute_quotes(
                    fair_value=fv_yes,
                    inventory_net=ctx.net_inventory,
                    gross_inventory=ctx.yes_shares + ctx.no_shares,
                    yes_token_id=ctx.yes_token_id,
                    no_token_id=ctx.no_token_id,
                    tick_size=tick,
                    spread_multiplier=_spread_mult,
                    yes_shares=ctx.yes_shares,
                    no_shares=ctx.no_shares,
                )

                # En WIND_DOWN : pas de nouveaux bids
                if _phase == "WIND_DOWN":
                    quote_set.yes_bid = None
                    quote_set.no_bid = None

                # GTD expiry pour sécurité crash (ordres meurent automatiquement)
                _gtd_expiry = _compute_gtd_expiry(ctx.expiry)

                # --- HEDGE CRITICAL → sell-only mode ---
                if _hedge_critical:
                    quote_set.yes_bid = None
                    quote_set.no_bid = None
                    if tick_count % 200 == 0:
                        print(f"!! HEDGE CRITICAL {ctx.strike_label}: sell-only mode (bids cancelled)")

                # --- CROSS-STRIKE PROTECTION ---
                # Si l'exposition totale ou directionnelle depasse la limite,
                # bloquer les achats pour eviter d'accumuler
                if cross_limit_breached:
                    quote_set.yes_bid = None
                    quote_set.no_bid = None

                # --- SELL PROTECTION : ne pas placer d'ASK sans token ---
                # Sur Polymarket, vendre YES/NO nécessite de posséder les tokens.
                # Sans inventaire, les SELL orders échouent avec "not enough balance".
                if ctx.yes_shares < 1:
                    quote_set.yes_ask = None
                if ctx.no_shares < 1:
                    quote_set.no_ask = None

                # --- UPDATE QUOTES ---
                quote_mgr.update_quotes(
                    market_key, quote_set, tick,
                    yes_market_bid=ctx.yes_bid, yes_market_ask=ctx.yes_ask,
                    no_market_bid=ctx.no_bid, no_market_ask=ctx.no_ask,
                    expiration=_gtd_expiry,
                )

                # --- SIM FILL CHECK (paper trading) ---
                if not cfg.LIVE_MODE:
                    quote_mgr.simulate_fills(
                        market_key,
                        yes_market_bid=ctx.yes_bid, yes_market_ask=ctx.yes_ask,
                        no_market_bid=ctx.no_bid, no_market_ask=ctx.no_ask,
                        yes_bid_vol=ctx.yes_bid_vol, yes_ask_vol=ctx.yes_ask_vol,
                        no_bid_vol=ctx.no_bid_vol, no_ask_vol=ctx.no_ask_vol,
                        imbalance=_imbalance,
                        realized_vol=rvol_calc.get_realized_vol(),
                        spread=ctx.yes_ask - ctx.yes_bid if ctx.yes_bid > 0 else 0.02,
                    )

                # --- PROCESS FILLS ---
                fills = quote_mgr.drain_fills()
                for fill in fills:
                    total_fills += 1
                    risk_mgr.record_fill()
                    cost = fill.price * fill.size
                    if fill.token_id == ctx.yes_token_id:
                        if fill.side == "BUY":
                            ctx.yes_shares += fill.size
                            ctx.yes_cost_basis += cost
                        else:
                            avg = _avg_cost(ctx.yes_shares, ctx.yes_cost_basis)
                            pnl = fill.size * (fill.price - avg)
                            realized_pnl += pnl
                            ctx.yes_cost_basis -= avg * fill.size
                            ctx.yes_shares = max(0, ctx.yes_shares - fill.size)
                            if ctx.yes_shares <= 0:
                                ctx.yes_cost_basis = 0.0
                    elif fill.token_id == ctx.no_token_id:
                        if fill.side == "BUY":
                            ctx.no_shares += fill.size
                            ctx.no_cost_basis += cost
                        else:
                            avg = _avg_cost(ctx.no_shares, ctx.no_cost_basis)
                            pnl = fill.size * (fill.price - avg)
                            realized_pnl += pnl
                            ctx.no_cost_basis -= avg * fill.size
                            ctx.no_shares = max(0, ctx.no_shares - fill.size)
                            if ctx.no_shares <= 0:
                                ctx.no_cost_basis = 0.0

                _total_unreal_risk = sum(m.unrealized_pnl for m in markets)
                risk_mgr.update_pnl(realized_pnl, unrealized_pnl=_total_unreal_risk)

                # --- POLL FILLS (actif, toutes les N secondes, par marché) ---
                now = time.time()
                if now - last_poll_time.get(market_key, 0) > cfg.MM_POLL_FILLS_INTERVAL:
                    quote_mgr.poll_fills(market_key)
                    last_poll_time[market_key] = now

                # --- RECONCILIATION ON-CHAIN (toutes les 5 min) ---
                now_rc = time.time()
                if cfg.LIVE_MODE and now_rc - last_reconcile_time > RECONCILE_INTERVAL:
                    reconcile_positions(markets, recorder)
                    last_reconcile_time = now_rc

                # --- LOGGING ---
                if tick_count % 50 == 0:
                    live_n = quote_mgr.get_live_count(market_key)
                    yb = quote_set.yes_bid
                    ya = quote_set.yes_ask
                    nb = quote_set.no_bid
                    na = quote_set.no_ask
                    yb_p = f"{yb.price:.3f}" if yb else "----"
                    ya_p = f"{ya.price:.3f}" if ya else "----"
                    nb_p = f"{nb.price:.3f}" if nb else "----"
                    na_p = f"{na.price:.3f}" if na else "----"
                    unreal = ctx.unrealized_pnl
                    total_unreal = sum(m.unrealized_pnl for m in markets)
                    hedge_str = f" | {hedge_mgr.get_log_string()}" if hedge_mgr else ""
                    _phase_str = market_phases.get(market_key, "Q")[0]  # Q/W/S
                    print(
                        f"MM {ctx.strike_label} BTC:{state.btc_price:.0f} IV:{vol*100:.1f}%{vol_src} "
                        f"FV:{fv_yes:.4f} Ctr:{quote_set.reservation_price:.4f} "
                        f"Sprd:{quote_set.optimal_spread:.4f} [{_phase_str}] "
                        f"Y[{yb_p}/{ya_p}] N[{nb_p}/{na_p}] "
                        f"q:{ctx.net_inventory:.0f} ord:{live_n} "
                        f"fills:{total_fills} rpnl:{realized_pnl:+.2f} upnl:{total_unreal:+.2f}"
                        f"{hedge_str}"
                    )

                # --- DATA LOG ---
                if tick_count % 10 == 0:
                    recorder.log_market_tick(
                        btc=state.btc_price, bid=ctx.yes_bid, ask=ctx.yes_ask,
                        bid_vol=ctx.yes_bid_vol, ask_vol=ctx.yes_ask_vol,
                        fv=fv_yes, edge=micro_adj, signal=None,
                        implied_vol=vol, spread=ctx.yes_ask - ctx.yes_bid,
                        mid_price=(ctx.yes_ask + ctx.yes_bid) / 2 if ctx.yes_bid > 0 else 0,
                        hours_left=hours_left,
                        bid_depth=ctx.yes_bid_depth, ask_depth=ctx.yes_ask_depth,
                        momentum=micro_momentum, spread_score=micro_spread_score, edge_adjustment=micro_adj,
                        market_strike=ctx.strike, kelly_size=cfg.MM_QUOTE_SIZE_USDC,
                        last_trade_price=ctx.last_trade_price,   # tape RL
                        last_trade_size=ctx.last_trade_size,     # tape RL
                        last_trade_side=ctx.last_trade_side,     # tape RL
                        funding_rate=state.funding_rate,         # funding rate BTC perp RL
                    )

            # --- DELTA HEDGE UPDATE (hors boucle for — une seule fois par tick) ---
            if hedge_mgr:
                hedge_mgr.update(markets, state.btc_price, vol)

            tick_count += 1
            await asyncio.sleep(0.1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nMM: Shutdown — cancel de tous les ordres...")
        quote_mgr.cancel_everything()
        if hedge_mgr:
            hedge_mgr.shutdown()
        print(f"MM: Session terminée — {total_fills} fills, P&L: {realized_pnl:.2f}$")


# --- BOUCLE SNIPER LEGACY ---

async def run_sniper(markets, state, recorder, micro, sniper):
    """Boucle principale mode Sniper (legacy v2)."""
    tick_count = 0
    while True:
        if state.btc_price <= 0:
            await asyncio.sleep(0.1)
            continue

        if micro:
            micro.update_btc(state.btc_price)

        for ctx in markets:
            if ctx.poly_ask <= 0:
                continue

            # --- VOLATILITÉ ---
            if cfg.VOL_SURFACE_ENABLED and state.vol_surface:
                vol = interpolate_iv(state.vol_surface, ctx.strike)
                if vol is None:
                    vol = state.implied_vol if state.implied_vol > 0 else cfg.VOLATILITY_FALLBACK
                vol_source = "S"
            elif state.implied_vol > 0:
                vol = state.implied_vol
                vol_source = ""
            else:
                vol = cfg.VOLATILITY_FALLBACK
                vol_source = "*"

            # --- SIGNAL BLACK-SCHOLES ---
            signal, edge, fv, hours_left = get_trading_signal(
                btc_price=state.btc_price,
                poly_bid=ctx.poly_bid,
                poly_ask=ctx.poly_ask,
                strike=ctx.strike,
                expiry_str=ctx.expiry,
                volatility=vol,
                min_edge=cfg.MIN_EDGE,
                side=cfg.SIDE_TO_WATCH
            )

            # --- MICROSTRUCTURE ---
            micro_signals = {"momentum": 0, "spread_score": 0, "edge_adjustment": 0, "imbalance": 0}
            if micro:
                spread = ctx.poly_ask - ctx.poly_bid
                micro_signals = micro.compute_signals(
                    ctx.poly_bid_vol, ctx.poly_ask_vol,
                    spread, state.btc_price, ctx.strike, cfg.SIDE_TO_WATCH
                )
                adjusted_edge = edge + micro_signals["edge_adjustment"]
                if adjusted_edge > cfg.MIN_EDGE and not signal:
                    signal = f"BUY_{cfg.SIDE_TO_WATCH}"
                elif adjusted_edge <= cfg.MIN_EDGE and signal:
                    signal = None
                edge = adjusted_edge

            # --- KELLY CRITERION ---
            kelly_size = cfg.TRADE_SIZE_USDC
            if cfg.KELLY_ENABLED and signal:
                kelly_size = compute_kelly_size(
                    fv, ctx.poly_ask,
                    bankroll=cfg.MAX_EXPOSURE_USDC,
                    kelly_fraction=cfg.KELLY_FRACTION,
                    min_size=cfg.MIN_TRADE_USDC,
                    max_size=cfg.TRADE_SIZE_USDC,
                    slippage_ticks=cfg.MM_SLIPPAGE_TICKS,
                    tick_size=_tick_size(ctx.poly_ask),
                )
                if kelly_size <= 0:
                    signal = None

            trade_size = kelly_size
            spread = ctx.poly_ask - ctx.poly_bid if ctx.poly_bid > 0 else 0
            mid_price = (ctx.poly_ask + ctx.poly_bid) / 2 if ctx.poly_bid > 0 else 0

            if tick_count % 10 == 0:
                recorder.log_market_tick(
                    btc=state.btc_price, bid=ctx.poly_bid, ask=ctx.poly_ask,
                    bid_vol=ctx.poly_bid_vol, ask_vol=ctx.poly_ask_vol,
                    fv=fv, edge=edge, signal=signal,
                    implied_vol=vol, spread=spread, mid_price=mid_price,
                    hours_left=hours_left,
                    bid_depth=ctx.bid_depth, ask_depth=ctx.ask_depth,
                    momentum=micro_signals["momentum"],
                    spread_score=micro_signals["spread_score"],
                    edge_adjustment=micro_signals["edge_adjustment"],
                    market_strike=ctx.strike, kelly_size=trade_size
                )

            pos_size = ctx.get_position_size()
            if tick_count % 50 == 0:
                pos_str = f" POS:{pos_size:.1f}" if pos_size > 0 else ""
                pend_str = " [ORD]" if ctx.pending_order else ""
                k_str = f" K:{trade_size:.1f}" if cfg.KELLY_ENABLED and signal else ""
                m_str = f" M:{micro_signals['edge_adjustment']:+.4f}" if micro else ""
                print(f"SN {ctx.strike_label} BTC:{state.btc_price:.0f} IV:{vol*100:.1f}%{vol_source} "
                      f"FV:{fv:.4f} B:{ctx.poly_bid:.3f} A:{ctx.poly_ask:.3f} E:{edge:+.3f} "
                      f"{signal or '-'}{pos_str}{pend_str}{k_str}{m_str}")

            # --- CANCEL & REPLACE ---
            if cfg.CANCEL_REPLACE_ENABLED and ctx.pending_order and sniper:
                optimal_price = compute_maker_buy_price(ctx.poly_bid, ctx.poly_ask)
                price_stale = abs(ctx.pending_order["price"] - optimal_price) > _tick_size(ctx.poly_bid)
                timed_out = (time.time() - ctx.pending_order["placed_at"]) > cfg.ORDER_TIMEOUT_SECONDS
                if price_stale or timed_out:
                    reason = "prix stale" if price_stale else "timeout"
                    if sniper.cancel_order(ctx.pending_order["order_id"]):
                        print(f"   C&R ({reason})")
                    ctx.pending_order = None

            # --- AUTO-SELL ---
            if pos_size > 0 and cfg.LIVE_MODE and sniper:
                should_sell = False
                sell_reason = ""
                if edge < -cfg.SELL_EDGE:
                    should_sell = True
                    sell_reason = f"Edge ({edge:+.3f})"
                elif hours_left < (cfg.SELL_TIME_SAFETY_MINUTES / 60):
                    should_sell = True
                    sell_reason = f"Temps ({hours_left:.1f}h)"
                if should_sell:
                    if ctx.pending_order and sniper:
                        sniper.cancel_order(ctx.pending_order["order_id"])
                        ctx.pending_order = None
                    token_id = ctx.yes_token_id if cfg.SIDE_TO_WATCH == "YES" else ctx.no_token_id
                    print(f"\n!! AUTO-SELL {ctx.strike_label} ! {sell_reason}")
                    result = sniper.execute_limit_sell(token_id, ctx.poly_bid, ctx.poly_ask, pos_size)
                    if result:
                        ctx.remove_position(token_id, pos_size)
                        await asyncio.sleep(10)
                    else:
                        await asyncio.sleep(5)
                    continue

            # --- ACHAT ---
            if signal and cfg.LIVE_MODE and sniper and not ctx.pending_order:
                total_positions = sum(1 for m in markets if m.get_position_size() > 0)
                total_exposure = sum(
                    m.get_position_size() * m.poly_ask for m in markets if m.poly_ask > 0
                )
                if total_positions >= cfg.MAX_OPEN_POSITIONS:
                    continue
                if total_exposure + trade_size > cfg.MAX_EXPOSURE_USDC:
                    continue

                token_id = ctx.yes_token_id if cfg.SIDE_TO_WATCH == "YES" else ctx.no_token_id
                print(f"\n!! SIGNAL {ctx.strike_label} {signal} ! Edge:{edge:+.3f} Size:${trade_size:.2f}")
                result = sniper.execute_limit_buy(token_id, ctx.poly_bid, ctx.poly_ask, trade_size)
                if result:
                    ctx.add_position(token_id, result["size"])
                    if cfg.CANCEL_REPLACE_ENABLED:
                        ctx.pending_order = {
                            "order_id": result["order_id"],
                            "price": result["price"],
                            "size": result["size"],
                            "placed_at": time.time()
                        }
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(5)

        tick_count += 1
        await asyncio.sleep(0.1)


# --- FUNDING RATE BINANCE (données RL) ---

async def fetch_binance_funding_rate(state):
    """
    Fetche le funding rate BTC perp Binance toutes les 60s.
    Proxy du sentiment directionnel du marché crypto — feature RL.
    Valeur typique : ±0.01% à ±0.1% par 8h (positif = marché long, négatif = bearish).
    """
    import httpx
    url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    state.funding_rate = float(data.get("lastFundingRate", 0.0))
        except Exception:
            pass  # Ne pas crasher si Binance fAPI est inaccessible
        await asyncio.sleep(60)


# --- MAIN ---

async def main():
    mode = "MM" if cfg.MM_ENABLED else "SNIPER"
    features = [mode]
    if cfg.MICROSTRUCTURE_ENABLED:
        features.append("Micro")
    if cfg.VOL_SURFACE_ENABLED:
        features.append("VolSurf")
    if not cfg.MM_ENABLED:
        if cfg.KELLY_ENABLED:
            features.append("Kelly")
        if cfg.CANCEL_REPLACE_ENABLED:
            features.append("C&R")
    if cfg.MULTI_STRIKE_ENABLED:
        features.append("MultiStrike")
    print(f"POLYBOT v3 | Mode: {mode} | Features: [{', '.join(features)}]")

    # 1. SETUP
    recorder = DataRecorder()

    if not os.path.exists(cfg.MARKET_DATA_FILE):
        print("!! Fichier de données absent. Lance 'tools/get_daily_id.py' d'abord !")
        return

    with open(cfg.MARKET_DATA_FILE, "r") as f:
        all_markets_data = json.load(f)

    if cfg.MULTI_STRIKE_ENABLED:
        markets_to_trade = all_markets_data
    else:
        markets_to_trade = [all_markets_data[0]]

    # Créer les contextes
    markets = []
    for m in markets_to_trade:
        ctx = MarketContext(m)
        ctx.load_positions_from_journal(recorder.trades_file)
        markets.append(ctx)
        print(f">> {m['question']} | Strike=${m['strike']:,.0f}")

    # État global
    state = BotState()
    state.expiry = markets[0].expiry

    # Microstructure
    micro = MicrostructureAnalyzer() if cfg.MICROSTRUCTURE_ENABLED else None

    # Connecteurs
    binance = BinanceConnector("btcusdt", state)
    asyncio.create_task(binance.connect())

    # PolyAdapters
    for ctx in markets:
        # YES book (toujours)
        yes_poly = PolyAdapterYES(ctx.yes_token_id, ctx)
        asyncio.create_task(yes_poly.connect())
        # NO book (seulement en mode MM)
        if cfg.MM_ENABLED:
            no_poly = PolyAdapterNO(ctx.no_token_id, ctx)
            asyncio.create_task(no_poly.connect())

    # Deribit
    num_strikes = cfg.VOL_SURFACE_STRIKES if cfg.VOL_SURFACE_ENABLED else 1
    deribit = DeribitConnector(state, num_strikes=num_strikes)
    asyncio.create_task(deribit.connect())

    # Funding rate Binance (données RL — polling 60s)
    asyncio.create_task(fetch_binance_funding_rate(state))

    print(f">> SYSTEME EN LIGNE — {len(markets)} marché(s)")

    # 2. BOUCLE
    if cfg.MM_ENABLED:
        await run_market_maker(markets, state, recorder, micro)
    else:
        sniper = Sniper(cfg.PRIVATE_KEY, recorder) if cfg.LIVE_MODE else None
        await run_sniper(markets, state, recorder, micro, sniper)


def _avg_cost(shares, cost_basis):
    """Retourne le coût moyen pondéré (VWAP) par share."""
    if shares <= 0:
        return 0.0
    return cost_basis / shares


if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n!! Arrêt manuel (CTRL+C).")
    finally:
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()
        except:
            pass
        print(">> Fermeture propre.")
