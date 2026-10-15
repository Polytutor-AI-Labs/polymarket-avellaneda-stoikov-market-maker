import asyncio
import json
import websockets

class BinanceConnector:
    def __init__(self, symbol, state_manager):
        self.symbol = symbol.lower()
        self.state = state_manager
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"

    async def connect(self):
        print(f"🔌 Binance: Connexion...")
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    print("✅ Binance Connecté")
                    async for message in ws:
                        data = json.loads(message)
                        price = float(data['p'])
                        self.state.update_btc(price)
            except:
                await asyncio.sleep(5)