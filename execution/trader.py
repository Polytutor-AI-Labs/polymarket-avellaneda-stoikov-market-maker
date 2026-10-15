from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
import config as cfg


def _tick_size(price):
    """Retourne le tick size Polymarket selon le prix."""
    if price < 0.10 or price > 0.90:
        return cfg.TICK_SIZE_EXTREME
    return cfg.TICK_SIZE_STANDARD


def compute_maker_buy_price(bid, ask):
    """bid + 1 tick (maker, 0% fees). Si ca crosserait le ask, on reste au bid."""
    tick = _tick_size(bid)
    maker_price = round(bid + tick, 3)
    if maker_price >= ask:
        return bid
    return maker_price


def compute_maker_sell_price(bid, ask):
    """ask - 1 tick (maker, 0% fees). Si ca crosserait le bid, on reste au ask."""
    tick = _tick_size(ask)
    maker_price = round(ask - tick, 3)
    if maker_price <= bid:
        return ask
    return maker_price


def create_clob_client(private_key, chain_id=137):
    """Crée et authentifie un ClobClient. Réutilisable par Sniper et QuoteManager."""
    temp = ClobClient("https://clob.polymarket.com", key=private_key, chain_id=chain_id)
    creds = temp.create_or_derive_api_creds()
    return ClobClient("https://clob.polymarket.com", key=private_key, chain_id=chain_id, creds=creds)


class Sniper:
    def __init__(self, private_key, recorder, chain_id=137):
        self.key = private_key
        self.chain_id = chain_id
        self.recorder = recorder
        self.client = None
        self._connect()

    def _connect(self):
        try:
            print("🔫 SNIPER: Génération des clés API (L2)...")
            self.client = create_clob_client(self.key, self.chain_id)
            print("✅ SNIPER: Armé et Authentifié (Ready to Fire).")
        except Exception as e:
            print(f"❌ SNIPER ERROR (Connexion): {e}")

    def execute_limit_buy(self, token_id, best_bid, best_ask, size_usdc):
        if not self.client or best_bid <= 0 or best_ask <= 0:
            print("⚠️ Tir annulé : Client non connecté ou prix invalide.")
            return False

        price = compute_maker_buy_price(best_bid, best_ask)
        size_shares = size_usdc / price

        if size_shares < 0.1:
            return False

        print(f"\n💥 FIRE (MAKER) ! Achat {size_shares:.2f} parts @ {price:.3f} (bid={best_bid:.3f} ask={best_ask:.3f})...")

        try:
            args = OrderArgs(
                price=price,
                size=size_shares,
                side="BUY",
                token_id=token_id
            )

            order = self.client.create_and_post_order(args)

            if order and order.get("success"):
                order_id = order.get('orderID')
                print(f"✅ SUCCÈS : Ordre {order_id} passé !")

                self.recorder.log_execution(
                    order_id=order_id,
                    side="BUY",
                    token_id=token_id,
                    price=price,
                    size_shares=size_shares,
                    status="OPEN"
                )
                return {"order_id": order_id, "price": price, "size": size_shares}
            else:
                print(f"⚠️ REJETÉ PAR LE MOTEUR : {order.get('errorMsg', order)}")
                return False

        except Exception as e:
            print(f"❌ ERREUR CRITIQUE TIR : {e}")
            return False

    def execute_limit_sell(self, token_id, best_bid, best_ask, size_shares):
        if not self.client or best_bid <= 0:
            print("⚠️ Vente annulée : Client non connecté ou prix invalide.")
            return False

        if size_shares < 0.1:
            return False

        price = compute_maker_sell_price(best_bid, best_ask)

        print(f"\n💸 SELL (MAKER) ! Vente {size_shares:.2f} parts @ {price:.3f} (bid={best_bid:.3f} ask={best_ask:.3f})...")

        try:
            args = OrderArgs(
                price=price,
                size=size_shares,
                side="SELL",
                token_id=token_id
            )

            order = self.client.create_and_post_order(args)

            if order and order.get("success"):
                order_id = order.get('orderID')
                print(f"✅ VENDU : Ordre {order_id} passé !")

                self.recorder.log_execution(
                    order_id=order_id,
                    side="SELL",
                    token_id=token_id,
                    price=price,
                    size_shares=size_shares,
                    status="CLOSED"
                )
                return {"order_id": order_id, "price": price, "size": size_shares}
            else:
                print(f"⚠️ REJETÉ PAR LE MOTEUR : {order.get('errorMsg', order)}")
                return False

        except Exception as e:
            print(f"❌ ERREUR CRITIQUE VENTE : {e}")
            return False

    def cancel_order(self, order_id):
        """Annule un ordre via le CLOB. Retourne True si succès."""
        if not self.client or not order_id:
            return False
        try:
            self.client.cancel(order_id)
            print(f"🔄 Ordre {order_id[:20]}... annulé")
            return True
        except Exception as e:
            print(f"⚠️ Échec cancel {order_id[:20]}... : {e}")
            return False
