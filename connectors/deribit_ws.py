import asyncio
import json
import websockets
from datetime import datetime

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _poly_expiry_to_deribit(expiry_str):
    """Convertit '2026-02-07T17:00:00Z' → '7FEB26'."""
    clean = expiry_str.replace("Z", "")
    dt = datetime.fromisoformat(clean)
    return f"{dt.day}{MONTHS[dt.month - 1]}{dt.year % 100}"


def _deribit_expiry_ts(instrument):
    """Extrait le timestamp d'expiry depuis un instrument Deribit."""
    return instrument.get("expiration_timestamp", 0)


class DeribitConnector:
    def __init__(self, state_manager, num_strikes=1):
        self.state = state_manager
        self.url = "wss://www.deribit.com/ws/api/v2"
        self.instrument_name = None       # ATM principal (rétro-compatible)
        self.instrument_names = []        # Multi-strike : liste d'instruments
        self.num_strikes = num_strikes    # Nombre de strikes à souscrire
        self._msg_id = 0

    def _next_id(self):
        self._msg_id += 1
        return self._msg_id

    async def _send(self, ws, method, params=None):
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params:
            msg["params"] = params
        await ws.send(json.dumps(msg))

    async def connect(self):
        print("🔌 Deribit: Connexion pour IV temps réel...")
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    # 1. Heartbeat
                    await self._send(ws, "public/set_heartbeat", {"interval": 30})
                    await ws.recv()  # ack

                    # 2. Discover & subscribe (multi-strike si activé)
                    instruments = await self._discover_instruments(ws)
                    if instruments:
                        self.instrument_names = instruments
                        self.instrument_name = instruments[0]  # ATM = premier
                        channels = [f"ticker.{name}.100ms" for name in instruments]
                        await self._send(ws, "public/subscribe", {"channels": channels})
                        strikes_str = ", ".join(n.split("-")[2] for n in instruments)
                        print(f"✅ Deribit: Abonné à {len(instruments)} strike(s) [{strikes_str}]")
                    else:
                        print("⚠️ Deribit: Aucun instrument trouvé, retry dans 60s...")
                        await asyncio.sleep(60)
                        continue

                    # 3. Listen loop
                    await self._listen(ws)

            except Exception as e:
                print(f"⚠️ Deribit reconnexion: {e}")
                await asyncio.sleep(5)

    async def _discover_instruments(self, ws):
        """
        Trouve les options BTC autour d'ATM pour notre expiry Polymarket.
        Retourne une liste d'instruments (1 si single, N si vol surface).
        """
        await self._send(ws, "public/get_instruments", {
            "currency": "BTC", "kind": "option", "expired": False
        })

        resp = json.loads(await ws.recv())
        instruments = resp.get("result", [])

        if not instruments:
            return None

        # Format cible Deribit
        target_expiry = _poly_expiry_to_deribit(self.state.expiry)

        # Filtrer par expiry exact (comparer le segment expiry, pas substring)
        # BTC-7FEB26-69000-C → split("-")[1] = "7FEB26"
        candidates = [i for i in instruments
                      if i["instrument_name"].split("-")[1] == target_expiry
                      and i.get("option_type") == "call"]

        # Si pas de match exact, trouver l'expiry la plus proche
        if not candidates:
            print(f"⚠️ Deribit: Pas d'expiry exacte {target_expiry}, recherche la plus proche...")
            try:
                clean = self.state.expiry.replace("Z", "")
                target_ts = datetime.fromisoformat(clean).timestamp() * 1000
            except:
                return None

            calls = [i for i in instruments if i.get("option_type") == "call"]
            if not calls:
                return None

            # Grouper par expiry, trouver la plus proche
            expiry_map = {}
            for i in calls:
                exp_ts = _deribit_expiry_ts(i)
                if exp_ts not in expiry_map:
                    expiry_map[exp_ts] = []
                expiry_map[exp_ts].append(i)

            closest_ts = min(expiry_map.keys(), key=lambda t: abs(t - target_ts))
            candidates = expiry_map[closest_ts]

            exp_name = candidates[0]["instrument_name"].split("-")[1] if candidates else "?"
            print(f"  → Utilise l'expiry {exp_name} ({len(candidates)} strikes)")

        # Trouver le strike ATM
        btc_price = self.state.btc_price
        if btc_price <= 0:
            strikes = [i["strike"] for i in candidates]
            btc_price = (min(strikes) + max(strikes)) / 2

        # Trier par distance à ATM
        candidates.sort(key=lambda i: abs(i["strike"] - btc_price))

        if self.num_strikes <= 1:
            # Mode classique : un seul strike ATM
            best = candidates[0]
            print(f"🎯 Deribit ATM: {best['instrument_name']} (strike={best['strike']:.0f}, BTC~{btc_price:.0f})")
            return [best["instrument_name"]]

        # Mode vol surface : N strikes les plus proches d'ATM
        selected = candidates[:self.num_strikes]
        selected.sort(key=lambda i: i["strike"])  # Trier par strike croissant
        names = [i["instrument_name"] for i in selected]
        print(f"🎯 Deribit Vol Surface: {len(selected)} strikes autour de BTC~{btc_price:.0f}")
        for i in selected:
            print(f"   → {i['instrument_name']} (strike={i['strike']:.0f})")
        return names

    async def _listen(self, ws):
        async for message in ws:
            try:
                data = json.loads(message)

                # Heartbeat: répondre aux test_request
                if data.get("method") == "heartbeat":
                    if data.get("params", {}).get("type") == "test_request":
                        await self._send(ws, "public/test")
                    continue

                # Ticker subscription update
                if data.get("method") == "subscription":
                    ticker = data.get("params", {}).get("data", {})
                    mark_iv = ticker.get("mark_iv", 0)
                    if mark_iv > 0:
                        iv_decimal = mark_iv / 100  # % → décimal

                        # Extraire le strike depuis l'instrument name
                        inst_name = ticker.get("instrument_name", "")
                        parts = inst_name.split("-")
                        if len(parts) >= 3:
                            try:
                                strike = float(parts[2])
                                self.state.vol_surface[strike] = iv_decimal
                            except ValueError:
                                pass

                        # Garder l'ATM comme implied_vol (rétro-compatible)
                        if inst_name == self.instrument_name:
                            self.state.implied_vol = iv_decimal

            except:
                pass
