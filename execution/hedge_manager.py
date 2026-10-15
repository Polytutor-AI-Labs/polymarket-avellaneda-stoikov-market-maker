"""
HedgeManager — Phase 2 : Exécution Hyperliquid.

Calcule le delta du portefeuille, exécute les rehedges via Hyperliquid
BTC perpetual futures dans un thread dédié (le SDK est synchrone),
réconcilie la position trackée vs réelle, et fournit un feedback
de santé au market maker (spread penalty / sell-only mode).

Journal séparé : data/hedge_journal.csv (ne pas mélanger avec trades_journal.csv).
"""

import csv
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone

from strategy.binary_delta import compute_portfolio_delta

logger = logging.getLogger(__name__)


class HedgeManager:
    """Gestionnaire de hedge avec exécution Hyperliquid."""

    def __init__(
        self,
        rebalance_interval=30.0,
        delta_tolerance=0.005,
        dry_run=True,
        hl_client=None,
    ):
        """
        Args:
            rebalance_interval: secondes entre chaque recalcul
            delta_tolerance: BTC de drift avant rehedge
            dry_run: si True, log only (Phase 1 compat)
            hl_client: instance de HyperliquidClient (None si dry_run)
        """
        self.rebalance_interval = rebalance_interval
        self.delta_tolerance = delta_tolerance
        self.dry_run = dry_run
        self.hl_client = hl_client

        # État interne — delta
        self._last_update = 0.0
        self._current_delta_btc = 0.0
        self._current_delta_usd = 0.0
        self._target_hedge_btc = 0.0
        self._cumulative_hedge_btc = 0.0
        self._drift_btc = 0.0
        self._rebalance_count = 0
        self._per_market = []

        # État interne — position Hyperliquid
        self._hl_position_btc = 0.0         # Position réelle sur Hyperliquid
        self._hl_unrealized_pnl = 0.0
        self._last_reconcile_time = 0.0
        self._reconcile_interval = 60.0     # Réconciliation toutes les 60s

        # Thread safety : protège _hl_position_btc entre l'event loop et le thread pool
        self._position_lock = threading.Lock()

        # État interne — trade async
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hedge")
        self._pending_trade: Future | None = None
        self._last_trade_time = 0.0
        self._last_trade_success = True

        # Santé
        self._last_successful_hedge_time = time.time()
        self._needs_hedge = False

        # Journal
        self._journal_path = os.path.join("data", "hedge_journal.csv")
        self._init_journal()

        # NOTE: La réconciliation initiale est faite de façon lazy (1ère exécution de _reconcile_position)
        # pour ne pas bloquer l'event loop asyncio au démarrage (appel HTTP synchrone risqué)

    def _init_journal(self):
        """Crée le fichier CSV de journal s'il n'existe pas."""
        if not os.path.exists(self._journal_path):
            os.makedirs(os.path.dirname(self._journal_path), exist_ok=True)
            with open(self._journal_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "action", "size_btc", "delta_before",
                    "delta_after", "hl_position_before", "hl_position_after",
                    "btc_price", "status", "detail",
                ])

    def update(self, markets, btc_price, implied_vol):
        """
        Recalcul périodique du delta. Appelé à chaque tick.

        Returns:
            True si un recalcul a eu lieu, False sinon
        """
        now = time.time()

        # Vérifier si un trade précédent est terminé
        self._check_pending_trade()

        if now - self._last_update < self.rebalance_interval:
            return False

        self._last_update = now

        # Calcul du delta portefeuille
        result = compute_portfolio_delta(markets, btc_price, implied_vol)
        self._current_delta_btc = result["total_delta_btc"]
        self._current_delta_usd = result["total_delta_usd"]
        self._per_market = result["per_market"]

        # Le hedge cible = opposé du delta (shorter pour neutraliser)
        self._target_hedge_btc = -self._current_delta_btc

        # Réconcilier la position réelle sur Hyperliquid
        if not self.dry_run and self.hl_client:
            self._reconcile_position()

        # Drift = écart entre hedge cible et position réelle
        if self.dry_run or not self.hl_client:
            # Mode dry-run : utiliser le hedge simulé cumulé
            self._drift_btc = self._target_hedge_btc - self._cumulative_hedge_btc
        else:
            # Mode live : lecture thread-safe de _hl_position_btc
            with self._position_lock:
                self._drift_btc = self._target_hedge_btc - self._hl_position_btc

        # Mémoriser si un hedge est nécessaire (utilisé par get_health())
        self._needs_hedge = abs(self._drift_btc) > self.delta_tolerance

        # Si le portefeuille est bien hedgé (drift dans la tolérance), tout va bien
        if not self._needs_hedge:
            self._last_successful_hedge_time = now

        # Rebalance si drift dépasse la tolérance
        if self._needs_hedge:
            # Ne pas lancer si un trade est déjà en cours
            if self._pending_trade is not None and not self._pending_trade.done():
                logger.debug("Hedge trade déjà en cours, skip")
                return True

            trade_btc = self._drift_btc
            self._rebalance_count += 1
            action = "SHORT" if trade_btc < 0 else "LONG"
            trade_usd = abs(trade_btc) * btc_price

            if self.dry_run or not self.hl_client:
                # Phase 1 : simulation
                self._cumulative_hedge_btc += trade_btc
                self._last_successful_hedge_time = now
                print(
                    f"HEDGE [DRY-RUN] #{self._rebalance_count}: "
                    f"{action} {abs(trade_btc):.6f} BTC (${trade_usd:.2f}) | "
                    f"delta={self._current_delta_btc:+.6f} BTC "
                    f"(${self._current_delta_usd:+.2f})"
                )
                self._log_journal(
                    action=action,
                    size_btc=abs(trade_btc),
                    delta_before=self._current_delta_btc,
                    hl_pos_before=0.0,
                    hl_pos_after=0.0,
                    btc_price=btc_price,
                    status="DRY-RUN",
                )
            else:
                # Phase 2 : exécution réelle dans un thread séparé
                is_buy = trade_btc > 0
                size = abs(trade_btc)
                with self._position_lock:
                    hl_pos_before = self._hl_position_btc

                print(
                    f"HEDGE SUBMIT #{self._rebalance_count}: "
                    f"{action} {size:.6f} BTC (${trade_usd:.2f}) | "
                    f"delta={self._current_delta_btc:+.6f} BTC "
                    f"hl_pos={self._hl_position_btc:+.6f}"
                )

                self._pending_trade = self._executor.submit(
                    self._execute_trade,
                    size, is_buy, btc_price, hl_pos_before,
                )

            # Log par marché si inventory non nul
            for m in self._per_market:
                if abs(m["net_inventory"]) > 0.5:
                    print(
                        f"   {m['strike_label']}: "
                        f"d/share={m['delta_unit']:.6f} "
                        f"q={m['net_inventory']:.0f} "
                        f"d_total={m['delta_btc']:+.6f} BTC"
                    )

        return True

    def _execute_trade(self, size_btc, is_buy, btc_price, hl_pos_before):
        """
        Exécute un trade sur Hyperliquid (appelé dans le thread pool).

        Returns:
            dict avec le résultat ou None
        """
        try:
            result = self.hl_client.place_market_order(size_btc, is_buy)
            if result and result.get("filled"):
                filled_size = result["size_btc"]
                action = "LONG" if is_buy else "SHORT"
                self._last_successful_hedge_time = time.time()
                self._last_trade_success = True
                self._last_trade_time = time.time()

                # Mettre à jour la position trackée (thread-safe)
                with self._position_lock:
                    if is_buy:
                        self._hl_position_btc += filled_size
                    else:
                        self._hl_position_btc -= filled_size

                self._log_journal(
                    action=action,
                    size_btc=filled_size,
                    delta_before=self._current_delta_btc,
                    hl_pos_before=hl_pos_before,
                    hl_pos_after=self._hl_position_btc,
                    btc_price=btc_price,
                    status="FILLED",
                )
                return result
            else:
                self._last_trade_success = False
                self._log_journal(
                    action="LONG" if is_buy else "SHORT",
                    size_btc=size_btc,
                    delta_before=self._current_delta_btc,
                    hl_pos_before=hl_pos_before,
                    hl_pos_after=self._hl_position_btc,
                    btc_price=btc_price,
                    status="FAILED",
                    detail="order returned None or not filled",
                )
                return None
        except Exception as e:
            self._last_trade_success = False
            logger.error(f"Hedge trade error: {e}")
            self._log_journal(
                action="LONG" if is_buy else "SHORT",
                size_btc=size_btc,
                delta_before=self._current_delta_btc,
                hl_pos_before=hl_pos_before,
                hl_pos_after=self._hl_position_btc,
                btc_price=btc_price,
                status="ERROR",
                detail=str(e)[:200],
            )
            return None

    def _check_pending_trade(self):
        """Vérifie si le trade soumis au thread pool est terminé."""
        if self._pending_trade is None:
            return
        if not self._pending_trade.done():
            return

        try:
            result = self._pending_trade.result(timeout=0)
            if result and result.get("filled"):
                with self._position_lock:
                    hl_pos = self._hl_position_btc
                print(f"HEDGE FILLED: {result.get('size_btc', '?')} BTC | hl_pos={hl_pos:+.6f}")
            else:
                print("HEDGE FAILED: ordre non rempli")
        except Exception as e:
            print(f"HEDGE ERROR: {e}")
        finally:
            self._pending_trade = None

    def _reconcile_position(self):
        """Synchronise la position trackée avec la position réelle Hyperliquid."""
        now = time.time()
        if now - self._last_reconcile_time < self._reconcile_interval:
            return

        self._last_reconcile_time = now

        if not self.hl_client:
            return

        pos = self.hl_client.get_position()
        if pos is None:
            logger.warning("Reconciliation Hyperliquid échouée (get_position=None)")
            return

        real_size = pos["size_btc"]
        drift = abs(real_size - self._hl_position_btc)

        if drift > 0.0001:
            logger.info(
                f"Hedge reconciliation: tracked={self._hl_position_btc:+.6f} "
                f"real={real_size:+.6f} drift={drift:.6f}"
            )
            with self._position_lock:
                self._hl_position_btc = real_size

        self._hl_unrealized_pnl = pos["unrealized_pnl"]

    def _log_journal(self, action, size_btc, delta_before, hl_pos_before,
                     hl_pos_after, btc_price, status, detail=""):
        """Écrit une ligne dans hedge_journal.csv."""
        try:
            with open(self._journal_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    action,
                    f"{size_btc:.6f}",
                    f"{delta_before:+.6f}",
                    f"{delta_before + (size_btc if action == 'LONG' else -size_btc):+.6f}",
                    f"{hl_pos_before:+.6f}",
                    f"{hl_pos_after:+.6f}",
                    f"{btc_price:.2f}",
                    status,
                    detail,
                ])
        except Exception as e:
            logger.error(f"Hedge journal write error: {e}")

    def get_health(self):
        """
        Feedback de santé pour le market maker.

        Returns:
            dict:
                - spread_penalty: multiplicateur de spread (1.0 = normal)
                - critical: True si mode sell-only
                - minutes_since_hedge: minutes depuis le dernier hedge réussi
                - hl_healthy: True si le client Hyperliquid fonctionne
        """
        # En dry-run, toujours sain
        if self.dry_run or not self.hl_client:
            return {
                "spread_penalty": 1.0,
                "critical": False,
                "minutes_since_hedge": 0.0,
                "hl_healthy": True,
            }

        minutes = (time.time() - self._last_successful_hedge_time) / 60.0
        hl_healthy = self.hl_client.is_healthy()

        if not hl_healthy or (self._needs_hedge and minutes > 15):
            # Critique : HL down, OU on doit hedger mais on n'y arrive pas depuis > 15 min
            return {
                "spread_penalty": 3.0,
                "critical": True,
                "minutes_since_hedge": minutes,
                "hl_healthy": hl_healthy,
            }
        elif minutes > 5:
            # Dégradé : 5-15 min sans hedge
            # Interpolation linéaire : 1.5x à 5min → 2.5x à 15min
            penalty = 1.5 + (minutes - 5) * (1.0 / 10.0)
            return {
                "spread_penalty": penalty,
                "critical": False,
                "minutes_since_hedge": minutes,
                "hl_healthy": hl_healthy,
            }
        else:
            # Normal : < 5 min
            return {
                "spread_penalty": 1.0,
                "critical": False,
                "minutes_since_hedge": minutes,
                "hl_healthy": hl_healthy,
            }

    def get_status(self):
        """Retourne l'état courant pour le logging."""
        with self._position_lock:
            hl_pos = self._hl_position_btc
        return {
            "delta_btc": self._current_delta_btc,
            "delta_usd": self._current_delta_usd,
            "target_hedge_btc": self._target_hedge_btc,
            "cumulative_hedge_btc": self._cumulative_hedge_btc,
            "drift_btc": self._drift_btc,
            "rebalance_count": self._rebalance_count,
            "per_market": self._per_market,
            "hl_position_btc": hl_pos,
            "hl_unrealized_pnl": self._hl_unrealized_pnl,
        }

    def get_log_string(self):
        """Résumé compact pour la ligne de log MM (toutes les 50 ticks)."""
        if self._current_delta_btc == 0 and self._rebalance_count == 0:
            return ""
        hl_str = ""
        if not self.dry_run and self.hl_client:
            with self._position_lock:
                hl_str = f" hl:{self._hl_position_btc:+.5f}"
        return (
            f"d:{self._current_delta_btc:+.5f}BTC "
            f"(${self._current_delta_usd:+.1f}) "
            f"hN:{self._rebalance_count}"
            f"{hl_str}"
        )

    def shutdown(self):
        """Arrêt propre : attend le trade en cours puis ferme l'executor."""
        print("HedgeManager: shutdown...")

        # Attendre le trade en cours
        if self._pending_trade is not None and not self._pending_trade.done():
            print("HedgeManager: attente du trade en cours...")
            try:
                self._pending_trade.result(timeout=10)
            except Exception as e:
                logger.error(f"HedgeManager shutdown trade error: {e}")

        # Fermer l'executor
        self._executor.shutdown(wait=True, cancel_futures=True)
        print("HedgeManager: shutdown complete")
