import yfinance as yf
import numpy as np
import pandas as pd
from optlib.market import TRADING_DAYS

def fetch_stock_data(ticker: str, period_years: int = 4):
    """
    Fetch historical data for a ticker and estimate mu/sigma.
    Returns (current_price, mu_hat, sigma_hat, price_series).
    """
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=period_years)

    data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    prices = data["Close"].squeeze()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    dt = 1 / TRADING_DAYS 
    mu_hat    = float(log_returns.mean() / dt)
    sigma_hat = float(log_returns.std()  / np.sqrt(dt))
    mu_hat = float(log_returns.mean() / dt + 0.5 * sigma_hat**2)
    current_price = float(prices.iloc[-1])
    return current_price, mu_hat, sigma_hat, prices


def rolling_mu_sigma(ticker: str, end_date: str, max_years: int = 10):
    """Rolling monthly mu/sigma estimates."""
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=max_years + 1)
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                       end=end_date, progress=False, auto_adjust=True)
    prices = data["Close"].squeeze()

    mu_list, sigma_list, periods = [], [], []
    current_start = pd.to_datetime(end_date) - pd.DateOffset(years=max_years)
    current_end   = pd.to_datetime(end_date)

    while current_start < current_end:
        window_data = prices.loc[current_start:current_end]
        if len(window_data) < 2:
            break
        log_r = np.log(window_data / window_data.shift(1)).dropna()
        dt = 1 / TRADING_DAYS 
        sigma_hat = float(log_r.std() / np.sqrt(dt))
        mu_hat = float(log_r.mean() / dt + 0.5 * sigma_hat**2)
        mu_list.append(mu_hat)
        sigma_list.append(sigma_hat)
        periods.append(current_start.strftime("%Y-%m"))
        current_start += pd.DateOffset(months=1)

    return periods, mu_list, sigma_list

def get_expiries(ticker: str) -> list:
    """Listed option expiry dates, e.g. ['2026-11-20', '2026-11-27', ...]."""
    return list(yf.Ticker(ticker).options)


def fetch_option_chain(ticker: str, expiry: str):
    """(spot, DataFrame of calls and puts for one expiry, with mid prices)."""
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    chain = tk.option_chain(expiry)
    frames = []
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()   # drop dead quotes
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["kind"] = kind
        frames.append(df[["strike", "bid", "ask", "mid", "impliedVolatility",
                          "volume", "openInterest", "kind"]])
    return spot, pd.concat(frames, ignore_index=True)