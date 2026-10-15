"""
HyperliquidClient — Wrapper autour du SDK hyperliquid-python-sdk.

Gère l'authentification, les ordres market IOC, la lecture de positions
et le contrôle de risque (double cap: par trade + position totale).
"""

import os
import time
import logging
from dotenv import load_dotenv

import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils.constants import MAINNET_API_URL

logger = logging.getLogger(__name__)

load_dotenv()


class HyperliquidClient:
    """Client Hyperliquid pour le hedge BTC perpetual futures."""

    def __init__(
        self,
        dry_run=True,
        max_position_btc=0.05,
        max_single_trade_btc=0.01,
        leverage=1,
        slippage=0.01,
    ):
        """
        Args:
            dry_run: si True, aucun ordre envoyé (log only)
            max_position_btc: position max autorisée en BTC
            max_single_trade_btc: taille max par rebalance
            leverage: levier (1 = pas de levier)
            slippage: slippage max pour les market orders (0.01 = 1%)
        """
        self.dry_run = dry_run
        self.max_position_btc = max_position_btc
        self.max_single_trade_btc = max_single_trade_btc
        self.leverage = leverage
        self.slippage = slippage

        # Compteur d'erreurs consécutives pour is_healthy()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        self._last_success_time = time.time()

        # Auth : wallet depuis .env
        pk = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        if not pk:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY absente du .env")
        if not pk.startswith("0x"):
            pk = "0x" + pk

        self._wallet = eth_account.Account.from_key(pk)
        self._address = self._wallet.address
        logger.info(f"Hyperliquid wallet: {self._address}")

        # Clients SDK
        self._info = Info(base_url=MAINNET_API_URL, skip_ws=True)
        self._exchange = None
        if not dry_run:
            self._exchange = Exchange(
                wallet=self._wallet,
                base_url=MAINNET_API_URL,
            )
            # Configurer le levier au démarrage
            try:
                self._exchange.update_leverage(
                    leverage=self.leverage,
                    name="BTC",
                    is_cross=True,
                )
                logger.info(f"Hyperliquid leverage set to {self.leverage}x cross")
            except Exception as e:
                logger.warning(f"Hyperliquid leverage setup failed: {e}")

    def get_position(self):
        """
        Position BTC perp actuelle.

        Returns:
            dict: {size_btc, entry_price, unrealized_pnl} ou None si pas de position
        """
        try:
            state = self._info.user_state(address=self._address)
            for pos_data in state.get("assetPositions", []):
                pos = pos_data.get("position", {})
                if pos.get("coin") == "BTC":
                    self._consecutive_errors = 0
                    self._last_success_time = time.time()
                    return {
                        "size_btc": float(pos.get("szi", 0)),
                        "entry_price": float(pos.get("entryPx", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                    }
            # Pas de position BTC = position nulle
            self._consecutive_errors = 0
            self._last_success_time = time.time()
            return {"size_btc": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0}
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"Hyperliquid get_position error: {e}")
            return None

    def get_balance(self):
        """
        Solde USD disponible (withdrawable).

        Returns:
            float ou None en cas d'erreur
        """
        try:
            state = self._info.user_state(address=self._address)
            balance = float(state.get("withdrawable", 0))
            self._consecutive_errors = 0
            self._last_success_time = time.time()
            return balance
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"Hyperliquid get_balance error: {e}")
            return None

    def place_market_order(self, size_btc, is_buy):
        """
        Market order IOC avec slippage cap et double safety.

        Args:
            size_btc: taille en BTC (toujours positive)
            is_buy: True = long, False = short

        Returns:
            dict: {filled, size_btc, is_buy, response} ou None en cas d'erreur
        """
        size_btc = abs(size_btc)

        # Arrondir à 4 décimales (0.0001 BTC = min Hyperliquid)
        # float_to_wire du SDK exige une représentation exacte → trop de décimales → crash
        size_btc = round(size_btc, 4)

        # Safety 1: cap par trade
        if size_btc > self.max_single_trade_btc:
            logger.warning(
                f"Trade capé: {size_btc:.6f} -> {self.max_single_trade_btc:.6f} BTC"
            )
            size_btc = self.max_single_trade_btc

        # Safety 2: vérifier la position projetée
        current_pos = self.get_position()
        if current_pos is not None:
            current_size = current_pos["size_btc"]
            if is_buy:
                projected = current_size + size_btc
            else:
                projected = current_size - size_btc

            if abs(projected) > self.max_position_btc:
                allowed = self.max_position_btc - abs(current_size)
                if allowed <= 0:
                    logger.warning(
                        f"Position max atteinte ({current_size:.6f} BTC), ordre refusé"
                    )
                    return None
                logger.warning(
                    f"Taille réduite pour respecter position max: "
                    f"{size_btc:.6f} -> {allowed:.6f} BTC"
                )
                size_btc = allowed

        if size_btc < 0.0001:
            logger.debug("Taille trop petite, ordre ignoré")
            return None

        # Dry-run : log sans exécuter
        if self.dry_run or self._exchange is None:
            action = "BUY" if is_buy else "SELL"
            logger.info(f"HEDGE [DRY-RUN] {action} {size_btc:.6f} BTC")
            self._consecutive_errors = 0
            self._last_success_time = time.time()
            return {
                "filled": True,
                "size_btc": size_btc,
                "is_buy": is_buy,
                "response": "dry-run",
            }

        # Exécution réelle avec retry ×3
        action = "BUY" if is_buy else "SELL"
        last_error = None
        for attempt in range(3):
            try:
                result = self._exchange.market_open(
                    name="BTC",
                    is_buy=is_buy,
                    sz=size_btc,
                    slippage=self.slippage,
                )

                # Vérifier le résultat
                status = result.get("status", "")
                if status == "ok":
                    self._consecutive_errors = 0
                    self._last_success_time = time.time()
                    # Extraire les fills
                    response_data = result.get("response", {})
                    statuses = response_data.get("data", {}).get("statuses", [])
                    filled_size = size_btc  # Par défaut, tout rempli
                    for s in statuses:
                        if "filled" in s:
                            filled_size = float(s["filled"].get("totalSz", size_btc))

                    logger.info(
                        f"HEDGE FILLED: {action} {filled_size:.6f} BTC "
                        f"(attempt {attempt + 1})"
                    )
                    return {
                        "filled": True,
                        "size_btc": filled_size,
                        "is_buy": is_buy,
                        "response": result,
                    }
                else:
                    last_error = f"status={status}: {result}"
                    logger.warning(
                        f"Hyperliquid order attempt {attempt + 1} failed: {last_error}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Hyperliquid order attempt {attempt + 1} error: {e}"
                )

            # Attendre avant retry
            if attempt < 2:
                time.sleep(1)

        self._consecutive_errors += 1
        logger.error(f"Hyperliquid order failed after 3 attempts: {last_error}")
        return None

    def close_position(self):
        """
        Ferme toute la position BTC perp.

        Returns:
            dict ou None en cas d'erreur
        """
        if self.dry_run or self._exchange is None:
            logger.info("HEDGE [DRY-RUN] close_position()")
            return {"filled": True, "response": "dry-run"}

        try:
            result = self._exchange.market_close(
                coin="BTC",
                slippage=self.slippage,
            )
            status = result.get("status", "")
            if status == "ok":
                self._consecutive_errors = 0
                self._last_success_time = time.time()
                logger.info("Hyperliquid: position BTC fermée")
                return {"filled": True, "response": result}
            else:
                self._consecutive_errors += 1
                logger.error(f"Hyperliquid close_position failed: {result}")
                return None
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"Hyperliquid close_position error: {e}")
            return None

    def is_healthy(self):
        """
        Vérifie la santé de la connexion.

        Returns:
            False après 3+ erreurs consécutives
        """
        return self._consecutive_errors < self._max_consecutive_errors

    def seconds_since_last_success(self):
        """Secondes depuis la dernière opération réussie."""
        return time.time() - self._last_success_time

    @property
    def address(self):
        return self._address
