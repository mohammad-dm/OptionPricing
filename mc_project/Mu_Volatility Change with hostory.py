import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_mu_sigma_rolling(ticker="SPY", end="2023-01-01", max_years=10):
    """
    Plot changes in mu_hat and sigma_hat using rolling windows.
    
    Parameters:
    - ticker: stock ticker string
    - end: end date for analysis
    - max_years: maximum lookback window in years
    """
    # Download historical data
    start_date = pd.to_datetime(end) - pd.DateOffset(years=max_years+1)  # add 1 extra year to ensure coverage
    data = yf.download(ticker, start=start_date, end=end)
    prices = data["Close"]

    mu_list = []
    sigma_list = []
    periods = []

    current_start = pd.to_datetime(end) - pd.DateOffset(years=max_years)
    current_end = pd.to_datetime(end)

    # Move start date in monthly steps
    while current_start < current_end:
        window_data = prices.loc[current_start:current_end]
        if len(window_data) < 2:
            break
        
        log_returns = np.log(window_data / window_data.shift(1)).dropna()
        dt = 1/252
        mu_hat = log_returns.mean() / dt
        sigma_hat = log_returns.std() / np.sqrt(dt)
        
        mu_list.append(mu_hat)
        sigma_list.append(sigma_hat)
        periods.append(current_start.strftime("%Y-%m"))
        
        # Move forward by 1 month
        current_start += pd.DateOffset(months=1)

    # Plot
    plt.figure(figsize=(14,7))
    plt.plot(periods, mu_list, marker='o', label="mu_hat")
    plt.plot(periods, sigma_list, marker='s', label="sigma_hat")
    plt.xticks(rotation=45)
    plt.ylabel("Estimate")
    plt.xlabel("Window Start Date")
    plt.title(f"Changes in mu_hat and sigma_hat ({ticker}) - rolling monthly over {max_years} years")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_mu_sigma_rolling(end="2023-01-01", max_years=10)