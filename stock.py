from typing import List, Dict, Any

class Instrument:
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price
        self.history = {} # {date: session_deal}
        self.session_deal = [] # [{"price", "amount"}]

    def add_session_deal(self, price_and_amount):
        self.session_deal.append(price_and_amount)

    def update_price(self, date):
        if len(self.session_deal) == 0:
            return
        # Simple VWAP or last price. Using last price for now.
        self.price = self.session_deal[-1]["price"]
        self.history[date] = self.session_deal
        self.session_deal.clear()

    def get_price(self):
        return self.price

class Stock(Instrument):
    def __init__(self, symbol: str, initial_price: float, initial_stock: int = 0):
        super().__init__(symbol, initial_price)
        self.initial_stock = initial_stock
        # Financial reports could be attached here if needed, 
        # but for NIFTY we might skip them or mock them.

class Option(Instrument):
    def __init__(self, symbol: str, price: float, strike: float, expiry: Any, type: str, underlying: str):
        super().__init__(symbol, price)
        self.strike = strike
        self.expiry = expiry
        self.type = type # "Call" or "Put"
        self.underlying = underlying
