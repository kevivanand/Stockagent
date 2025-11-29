import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import List, Dict, Any

class DataLoader:
    def __init__(self):
        self.cache = {}

    def fetch_nifty_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches NIFTY 50 data from yfinance.
        Symbol for NIFTY 50 in yfinance is '^NSEI'.
        """
        ticker = "^NSEI"
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data fetched for NIFTY 50. Check internet connection or date range.")
        
        # Ensure index is datetime
        data.index = pd.to_datetime(data.index)
        return data

    def generate_option_chain(self, underlying_price: float, date: datetime, num_strikes: int = 5) -> List[Dict[str, Any]]:
        """
        Simulates an option chain for NIFTY based on the underlying price.
        Generates strikes around the ATM (At The Money) price.
        """
        underlying_price = float(underlying_price)
        atm_strike = round(underlying_price / 50) * 50
        strikes = []
        for i in range(-num_strikes // 2 + 1, num_strikes // 2 + 1):
            strikes.append(atm_strike + i * 50)
        
        # Assume weekly expiry (next Thursday)
        days_ahead = 3 - date.weekday()
        if days_ahead <= 0: 
            days_ahead += 7
        expiry_date = date + timedelta(days=days_ahead)
        
        options = []
        risk_free_rate = 0.07 # 7% India risk free rate approx
        volatility = 0.15 # 15% IV assumption
        
        time_to_expiry = days_ahead / 365.0
        
        for strike in strikes:
            # Calculate Call Price (Black-Scholes)
            d1 = (np.log(underlying_price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            call_price = underlying_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            
            # Calculate Put Price (Put-Call Parity)
            put_price = call_price + strike * np.exp(-risk_free_rate * time_to_expiry) - underlying_price
            
            options.append({
                "symbol": f"NIFTY{expiry_date.strftime('%d%b').upper()}{strike}CE",
                "type": "Call",
                "strike": strike,
                "expiry": expiry_date,
                "price": round(max(0.05, call_price), 2),
                "underlying": "NIFTY"
            })
            
            options.append({
                "symbol": f"NIFTY{expiry_date.strftime('%d%b').upper()}{strike}PE",
                "type": "Put",
                "strike": strike,
                "expiry": expiry_date,
                "price": round(max(0.05, put_price), 2),
                "underlying": "NIFTY"
            })
            
        return options
