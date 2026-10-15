class MarketData:
    def __init__(self):
        self.btc_price = 0.0
        self.poly_bid = 0.0
        self.poly_ask = 0.0
        self.bids = []
        self.asks = []

class StateManager:
    def __init__(self):
        self.data = MarketData()

    def update_btc(self, price):
        self.data.btc_price = price

    def update_poly(self, bid, ask):
        self.data.poly_bid = bid
        self.data.poly_ask = ask

    def update_poly_book(self, bids, asks):
        self.data.bids = bids
        self.data.asks = asks
        if bids: self.data.poly_bid = float(bids[0].get('price', 0))
        if asks: self.data.poly_ask = float(asks[0].get('price', 0))