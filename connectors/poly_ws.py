import asyncio
import json
import websockets
import httpx
import time
import os
from dotenv import load_dotenv

load_dotenv()

class PolyConnector:
    def __init__(self, asset_id: str, state_manager):
        self.asset_id = asset_id
        self.state = state_manager
        self.url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.base_url = "https://clob.polymarket.com"
        
        self._last_trade_cache = (0.0, 0.0)
        self._last_trade_fetch = 0

    async def connect(self):
        print(f"🔌 PolyBot : Connexion au flux...")
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    print("✅ Poly WS Connecté")
                    
                    # On s'abonne uniquement au market (puisque le trade est dedans)
                    sub_msg = {"assets_ids": [self.asset_id], "type": "market"}
                    await ws.send(json.dumps(sub_msg))
                    
                    await self._listen(ws)
            except Exception as e:
                print(f"⚠️ Reconnexion WS: {e}")
                await asyncio.sleep(5)

    async def _listen(self, ws):
        async for message in ws:
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for item in data: self._process(item)
                else: self._process(data)
            except: pass

    async def get_last_trade(self, token_id: str, debug: bool = False):
        # Priorité au cache du WebSocket (qui contient le volume)
        current_time = time.time()
        
        # Si le cache est récent (5 secondes), on l'utilise
        if (current_time - self._last_trade_fetch) < 5.0:
            return self._last_trade_cache
            
        # Sinon, on retourne le vieux cache (mieux que rien)
        return self._last_trade_cache

    def _process(self, data):
        event_type = data.get("event_type")
        
        # --- 1. Gestion du Carnet d'ordres ---
        if event_type == "book":
            bids = sorted(data.get("bids", []), key=lambda x: float(x.get("price", 0)), reverse=True)
            asks = sorted(data.get("asks", []), key=lambda x: float(x.get("price", 999)))
            self.state.update_poly_book(bids, asks)
            
        # --- 2. Gestion des Prix ET DU VOLUME (C'est ici la correction) ---
        elif event_type == "price_change":
            # Le log a montré que la clé est "price_changes" (pluriel)
            changes = data.get("price_changes") or data.get("changes") or []
            
            for c in changes:
                p = float(c.get("price", 0))
                s = float(c.get("size", 0)) # LE FAMEUX VOLUME !
                side = c.get("side")

                # Mise à jour du State (Prix Live)
                if side == "BUY": self.state.update_poly(p, self.state.data.poly_ask)
                else: self.state.update_poly(self.state.data.poly_bid, p)

                # Si on détecte du volume, on met à jour le cache TRADE
                if s > 0:
                    self._last_trade_cache = (p, s)
                    self._last_trade_fetch = time.time()
                    print(f"💰 VOLUME DÉTECTÉ : {p}$ x {s}") # Petit print de confirmation