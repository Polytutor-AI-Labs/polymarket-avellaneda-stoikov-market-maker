"""
Quote Manager : gestion du cycle de vie des ordres market maker sur Polymarket CLOB.

Responsabilités :
1. Placer des quotes (bid + ask sur YES + NO) = 4 ordres par marché
2. Cancel les quotes stale quand le prix drift > seuil
3. Détecter les fills via get_order() polling
4. Cancel & requote sur cycle de refresh
5. Tracker tous les ordres live par marché
"""

import time
import csv
import os
import random
from dataclasses import dataclass, field
from py_clob_client.clob_types import OrderArgs, OrderType
from execution.trader import _tick_size
import config as cfg
from strategy.fill_model import fill_probability_v1, fill_probability_v2


@dataclass
class LiveOrder:
    """Ordre live sur le CLOB."""
    order_id: str
    token_id: str
    side: str           # "BUY" ou "SELL"
    price: float
    size: float
    label: str          # "YES_BID", "YES_ASK", "NO_BID", "NO_ASK"
    placed_at: float
    already_matched: float = 0.0  # Cumul déjà comptabilisé (évite double-count)


@dataclass
class MarketOrders:
    """Tous les ordres live pour un marché."""
    yes_bid: LiveOrder | None = None
    yes_ask: LiveOrder | None = None
    no_bid: LiveOrder | None = None
    no_ask: LiveOrder | None = None


@dataclass
class FillEvent:
    """Un fill détecté."""
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    label: str
    filled_at: float


class QuoteManager:
    def __init__(self, client, recorder,
                 requote_interval=5.0,
                 price_drift_ticks=1,
                 order_timeout=30.0,
                 live_mode=True):
        self.client = client
        self.recorder = recorder
        self.requote_interval = requote_interval
        self.price_drift_ticks = price_drift_ticks
        self.order_timeout = order_timeout
        self.live_mode = live_mode

        self.live_orders = {}      # {market_key: MarketOrders}
        self.last_requote = {}     # {market_key: timestamp}
        self.recent_fills = []     # List[FillEvent]
        self.stats = {"placed": 0, "cancelled": 0, "fills": 0, "errors": 0}
        self._error_cooldown = {}  # {label: timestamp} — supprime le spam d'erreurs répétées
        self._last_sim_fill_check = 0
        self._sim_rng = random.Random(42)

    def update_quotes(self, market_key, quote_set, tick_size,
                      yes_market_bid=0, yes_market_ask=0,
                      no_market_bid=0, no_market_ask=0,
                      expiration=None):
        """
        Point d'entrée principal : met à jour les ordres live en fonction
        des quotes désirées calculées par AS-lite.
        Clamp les prix pour rester PASSIF (maker) : bid < market ask, ask > market bid.
        expiration: timestamp unix (secondes) pour GTD, None = GTC.
        """
        now = time.time()
        if market_key not in self.live_orders:
            self.live_orders[market_key] = MarketOrders()
            self.last_requote[market_key] = 0

        # Clamp les quotes pour ne jamais croiser le spread (rester maker)
        self._clamp_quotes(quote_set, tick_size,
                           yes_market_bid, yes_market_ask,
                           no_market_bid, no_market_ask)

        current = self.live_orders[market_key]

        # Tolerance: 2 ticks (spec v1.0 — préserve la queue priority)
        tolerance_ticks = getattr(cfg, 'MM_ORDER_TOLERANCE_TICKS', 2)

        slots = [
            ("yes_bid", quote_set.yes_bid),
            ("yes_ask", quote_set.yes_ask),
            ("no_bid", quote_set.no_bid),
            ("no_ask", quote_set.no_ask),
        ]

        to_cancel = []
        to_place = []

        for slot_name, desired in slots:
            live = getattr(current, slot_name)

            if desired is None:
                # Pull ce côté
                if live is not None:
                    to_cancel.append((slot_name, live))
                continue

            if live is None:
                # Pas d'ordre live -> placer
                to_place.append((slot_name, desired))
                continue

            # Ordre live existe -> cancel/replace uniquement si :
            # 1. Prix drifté >= 2 ticks (quote stale)
            # 2. Timeout (ordre approche de son expiry GTD → 90s)
            price_drift = abs(live.price - desired.price) / tick_size
            timed_out = (now - live.placed_at) > self.order_timeout

            if price_drift >= tolerance_ticks or timed_out:
                to_cancel.append((slot_name, live))
                to_place.append((slot_name, desired))

        # --- Cancel ---
        if to_cancel:
            self._cancel_orders(to_cancel, current)

        # --- Place ---
        for slot_name, desired in to_place:
            result = self._place_order(desired, expiration=expiration)
            if result:
                setattr(current, slot_name, result)

    def _clamp_quotes(self, quote_set, tick_size,
                      yes_market_bid, yes_market_ask,
                      no_market_bid, no_market_ask):
        """
        Clamp les prix pour rester PASSIF (maker, 0% fees).
        - BID doit être < market best ask (sinon on cross et on est taker)
        - ASK doit être > market best bid (sinon on cross et on est taker)
        Si le clamp rend le quote impossible, on le supprime (None).
        """
        if quote_set.yes_bid and yes_market_ask > 0:
            max_bid = round(yes_market_ask - tick_size, 4)
            if quote_set.yes_bid.price >= yes_market_ask:
                quote_set.yes_bid.price = max_bid
            if max_bid < tick_size:
                quote_set.yes_bid = None

        if quote_set.yes_ask and yes_market_bid > 0:
            min_ask = round(yes_market_bid + tick_size, 4)
            if quote_set.yes_ask.price <= yes_market_bid:
                quote_set.yes_ask.price = min_ask
            if min_ask > 1.0 - tick_size:
                quote_set.yes_ask = None

        if quote_set.no_bid and no_market_ask > 0:
            max_bid = round(no_market_ask - tick_size, 4)
            if quote_set.no_bid.price >= no_market_ask:
                quote_set.no_bid.price = max_bid
            if max_bid < tick_size:
                quote_set.no_bid = None

        if quote_set.no_ask and no_market_bid > 0:
            min_ask = round(no_market_bid + tick_size, 4)
            if quote_set.no_ask.price <= no_market_bid:
                quote_set.no_ask.price = min_ask
            if min_ask > 1.0 - tick_size:
                quote_set.no_ask = None

    def _cancel_orders(self, cancel_list, current_orders):
        """
        Cancel les ordres et detecte les fills.
        Sequence : poll AVANT cancel, cancel, poll APRES cancel (rattrape la race condition).
        """
        if not self.live_mode:
            for slot_name, order in cancel_list:
                print(f"   [SIM] Cancel {order.label} {order.order_id[:16]}...")
                setattr(current_orders, slot_name, None)
            return

        # 1. Poll AVANT cancel pour detecter les fills en cours
        for slot_name, order in cancel_list:
            self._poll_single_order(order)

        # 2. Cancel batch
        ids = [o.order_id for _, o in cancel_list]
        try:
            self.client.cancel_orders(ids)
            self.stats["cancelled"] += len(ids)
        except Exception:
            # Fallback : cancel un par un
            for slot_name, order in cancel_list:
                try:
                    self.client.cancel(order.order_id)
                    self.stats["cancelled"] += 1
                except Exception:
                    pass  # Deja fill ou deja cancelled

        # 3. Poll APRES cancel pour rattraper les fills entre poll et cancel
        for slot_name, order in cancel_list:
            self._poll_single_order(order)

        # 4. Clear les slots
        for slot_name, order in cancel_list:
            setattr(current_orders, slot_name, None)

    def _poll_single_order(self, order):
        """Poll un seul ordre pour detecter les fills incrementaux."""
        try:
            api_order = self.client.get_order(order.order_id)
            if not api_order:
                return
            size_matched = float(api_order.get("size_matched", 0))
            new_fill = size_matched - order.already_matched
            # Monotonicity check : ignorer si negatif (donnee stale/glitch)
            if new_fill > 0.01:
                self._record_fill(order, new_fill)
                order.already_matched = size_matched
        except Exception:
            pass

    def _place_order(self, quote, expiration=None):
        """
        Place un ordre sur le CLOB. Retourne LiveOrder ou None.
        expiration: timestamp unix (secondes) pour GTD. None = GTC.
        GTD garantit que l'ordre meurt automatiquement si le bot crashe.
        """
        # Polymarket impose un minimum de 5 shares par ordre
        if quote.size < 5.0:
            return None

        if not self.live_mode:
            order_type_str = f"GTD@{expiration}" if expiration else "GTC"
            print(f"   [SIM] Place {quote.label} {quote.side} {quote.size:.1f} @ {quote.price:.3f} [{order_type_str}]")
            return LiveOrder(
                order_id=f"SIM_{quote.label}_{time.time():.0f}",
                token_id=quote.token_id,
                side=quote.side,
                price=quote.price,
                size=quote.size,
                label=quote.label,
                placed_at=time.time(),
            )

        try:
            args = OrderArgs(
                price=quote.price,
                size=quote.size,
                side=quote.side,
                token_id=quote.token_id,
                expiration=expiration if expiration else 0,
            )
            order_type = OrderType.GTD if expiration else OrderType.GTC
            result = self.client.create_and_post_order(args, order_type=order_type)

            if result and result.get("success"):
                order_id = result.get("orderID", "")
                self.stats["placed"] += 1
                # MM_LIVE n'est PAS loggué au journal (spam: cancel/replace toutes les 5s)
                # Seuls les fills (MM_FILL_BUY/SELL) sont loggués via _record_fill()
                return LiveOrder(
                    order_id=order_id,
                    token_id=quote.token_id,
                    side=quote.side,
                    price=quote.price,
                    size=quote.size,
                    label=quote.label,
                    placed_at=time.time(),
                )
            else:
                err = result.get("errorMsg", str(result)) if result else "no response"
                print(f"   !! Rejet {quote.label}: {err}")
                self.stats["errors"] += 1
                return None

        except Exception as e:
            now = time.time()
            last_err = self._error_cooldown.get(quote.label, 0)
            if now - last_err > 60:  # Log l'erreur max 1x/minute par label
                print(f"   !! Erreur placement {quote.label}: {e}")
                self._error_cooldown[quote.label] = now
            self.stats["errors"] += 1
            return None

    def simulate_fills(self, market_key,
                        yes_market_bid=0, yes_market_ask=0,
                        no_market_bid=0, no_market_ask=0,
                        yes_bid_vol=0, yes_ask_vol=0,
                        no_bid_vol=0, no_ask_vol=0,
                        imbalance=0.0, realized_vol=None,
                        spread=0.02):
        """
        Simule les fills en paper trading : vérifie si le marché a traversé nos prix.
        Rate-limité à SIM_FILL_CHECK_INTERVAL secondes.
        """
        if self.live_mode:
            return

        now = time.time()
        if now - self._last_sim_fill_check < cfg.SIM_FILL_CHECK_INTERVAL:
            return
        self._last_sim_fill_check = now

        current = self.live_orders.get(market_key)
        if not current:
            return

        use_stochastic = (cfg.SIM_FILL_MODEL == "stochastic")

        # YES BID: fill si yes_market_ask <= order.price (quelqu'un vend au prix de notre bid)
        if current.yes_bid is not None and yes_market_ask > 0:
            order = current.yes_bid
            if yes_market_ask <= order.price:
                do_fill = True
                if use_stochastic:
                    fp = fill_probability_v2(
                        order.size, yes_ask_vol, imbalance, is_buy=True,
                        spread=spread, realized_vol=realized_vol,
                    )
                    do_fill = self._sim_rng.random() < fp
                if do_fill:
                    self._record_fill(order, order.size)
                    current.yes_bid = None

        # YES ASK: fill si yes_market_bid >= order.price (quelqu'un achète au prix de notre ask)
        if current.yes_ask is not None and yes_market_bid > 0:
            order = current.yes_ask
            if yes_market_bid >= order.price:
                do_fill = True
                if use_stochastic:
                    fp = fill_probability_v2(
                        order.size, yes_bid_vol, imbalance, is_buy=False,
                        spread=spread, realized_vol=realized_vol,
                    )
                    do_fill = self._sim_rng.random() < fp
                if do_fill:
                    self._record_fill(order, order.size)
                    current.yes_ask = None

        # NO BID: fill si no_market_ask <= order.price
        if current.no_bid is not None and no_market_ask > 0:
            order = current.no_bid
            if no_market_ask <= order.price:
                do_fill = True
                if use_stochastic:
                    fp = fill_probability_v2(
                        order.size, no_ask_vol, imbalance, is_buy=True,
                        spread=spread, realized_vol=realized_vol,
                    )
                    do_fill = self._sim_rng.random() < fp
                if do_fill:
                    self._record_fill(order, order.size)
                    current.no_bid = None

        # NO ASK: fill si no_market_bid >= order.price
        if current.no_ask is not None and no_market_bid > 0:
            order = current.no_ask
            if no_market_bid >= order.price:
                do_fill = True
                if use_stochastic:
                    fp = fill_probability_v2(
                        order.size, no_bid_vol, imbalance, is_buy=False,
                        spread=spread, realized_vol=realized_vol,
                    )
                    do_fill = self._sim_rng.random() < fp
                if do_fill:
                    self._record_fill(order, order.size)
                    current.no_ask = None

    def poll_fills(self, market_key):
        """
        Polling actif : vérifie le statut de chaque ordre live via l'API.
        Détecte les fills partiels ou complets.
        Utilise already_matched pour éviter le double-count (size_matched est cumulatif).
        """
        current = self.live_orders.get(market_key)
        if not current or not self.live_mode:
            return

        for slot_name in ["yes_bid", "yes_ask", "no_bid", "no_ask"]:
            order = getattr(current, slot_name)
            if order is None:
                continue

            try:
                api_order = self.client.get_order(order.order_id)
                if not api_order:
                    continue

                size_matched = float(api_order.get("size_matched", 0))
                original_size = float(api_order.get("original_size", order.size))

                # Calcul incremental : seulement le NOUVEAU fill depuis le dernier poll
                new_fill = size_matched - order.already_matched

                # Monotonicity check : ignorer si negatif (donnee stale)
                if new_fill < 0:
                    continue

                if size_matched >= original_size * 0.95:
                    # Full fill — enregistrer le reste non comptabilise
                    if new_fill > 0.01:
                        self._record_fill(order, new_fill)
                    order.already_matched = size_matched
                    setattr(current, slot_name, None)
                elif new_fill > 0.01:
                    # Partial fill — enregistrer seulement l'increment
                    self._record_fill(order, new_fill)
                    order.already_matched = size_matched

            except Exception:
                pass  # API error, skip

    def _record_fill(self, order, size=None):
        """Enregistre un fill détecté."""
        fill_size = size if size else order.size
        fill = FillEvent(
            order_id=order.order_id,
            token_id=order.token_id,
            side=order.side,
            price=order.price,
            size=fill_size,
            label=order.label,
            filled_at=time.time(),
        )
        self.recent_fills.append(fill)
        self.stats["fills"] += 1
        print(f"   >> FILL {fill.label} {fill.side} {fill_size:.1f} @ {fill.price:.3f}")

        # Log dans le journal
        if self.live_mode:
            status = "MM_FILL_BUY" if order.side == "BUY" else "MM_FILL_SELL"
            self.recorder.log_execution(
                order_id=order.order_id,
                side=order.side,
                token_id=order.token_id,
                price=order.price,
                size_shares=fill_size,
                status=status
            )
        elif cfg.SIM_FILL_LOG_TO_CSV:
            status = "SIM_FILL_BUY" if order.side == "BUY" else "SIM_FILL_SELL"
            self.recorder.log_execution(
                order_id=order.order_id,
                side=order.side,
                token_id=order.token_id,
                price=order.price,
                size_shares=fill_size,
                status=status
            )

    def drain_fills(self):
        """Retourne et vide les fills récents pour traitement par l'inventaire."""
        fills = self.recent_fills.copy()
        self.recent_fills.clear()
        return fills

    def cancel_all_market(self, market_key):
        """Cancel tous les ordres d'un marché (circuit breaker)."""
        current = self.live_orders.get(market_key)
        if not current:
            return

        ids = []
        for slot in ["yes_bid", "yes_ask", "no_bid", "no_ask"]:
            order = getattr(current, slot)
            if order is not None:
                ids.append(order.order_id)

        if ids and self.live_mode:
            try:
                self.client.cancel_orders(ids)
            except Exception:
                try:
                    self.client.cancel_all()
                except Exception as e:
                    print(f"   !! cancel_all échoué: {e}")

        self.live_orders[market_key] = MarketOrders()

    def cancel_everything(self):
        """Nuclear : cancel tous les ordres sur tous les marchés."""
        if self.live_mode:
            try:
                self.client.cancel_all()
                print("   !! CANCEL ALL envoyé")
            except Exception as e:
                print(f"   !! cancel_all échoué: {e}")
        self.live_orders.clear()

    def get_live_count(self, market_key=None):
        """Nombre d'ordres live."""
        if market_key:
            current = self.live_orders.get(market_key)
            if not current:
                return 0
            return sum(1 for s in ["yes_bid", "yes_ask", "no_bid", "no_ask"]
                       if getattr(current, s) is not None)
        return sum(self.get_live_count(mk) for mk in self.live_orders)

    def cancel_side(self, market_key, side):
        """
        Cancel tous les ordres d'un côté (BUY ou SELL) pour un marché.
        Utilisé pour hard inventory limits.

        Args:
            market_key: Marché cible
            side: "BUY" ou "SELL"
        """
        current = self.live_orders.get(market_key)
        if not current:
            return

        ids = []
        slots_to_clear = []

        for slot in ["yes_bid", "yes_ask", "no_bid", "no_ask"]:
            order = getattr(current, slot)
            if order is not None and order.side == side:
                ids.append(order.order_id)
                slots_to_clear.append(slot)

        if ids and self.live_mode:
            try:
                self.client.cancel_orders(ids)
                self.stats["cancelled"] += len(ids)
                print(f"   → Cancelled {len(ids)} {side} orders on {market_key}")
            except Exception as e:
                print(f"   !! cancel_side failed: {e}")

        # Clear les slots
        for slot in slots_to_clear:
            setattr(current, slot, None)

    def post_market_order(self, token_id, side, price, size):
        """
        Poste un ordre market (immediate-or-cancel like) pour panic close.
        Utilise un prix agressif pour execution immédiate.

        Args:
            token_id: Token ID
            side: "BUY" ou "SELL"
            price: Prix du market (bid si SELL, ask si BUY)
            size: Taille en shares
        """
        if not self.live_mode or not self.client:
            print(f"   (dry-run) MARKET {side} {size:.1f} @ {price:.3f}")
            return

        if size < 5:  # Polymarket min size
            print(f"   !! Size {size:.1f} < 5 shares, skip market order")
            return

        try:
            args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id
            )
            result = self.client.create_and_post_order(args)
            if result and result.get("success"):
                order_id = result.get("orderID")
                print(f"   ✅ MARKET {side} posted: {order_id[:16]}...")
                self.stats["placed"] += 1
                # Log dans le recorder
                self.recorder.log_execution(
                    order_id=order_id,
                    side=side,
                    token_id=token_id,
                    price=price,
                    size_shares=size,
                    status="MM_PANIC_CLOSE"
                )
            else:
                print(f"   !! Market order rejected: {result.get('errorMsg', result)}")
                self.stats["errors"] += 1
        except Exception as e:
            print(f"   !! Market order failed: {e}")
            self.stats["errors"] += 1
