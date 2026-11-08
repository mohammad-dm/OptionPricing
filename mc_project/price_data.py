# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:16:06 2026

@author: model
"""

#%matplotlib
import yfinance as yf 
import numpy as np
import matplotlib.pyplot as plt 


def stock_data ():
    ticker = "SPY"
    data = yf.download(ticker, start="2018-01-01", end="2022-01-04")
    
    prices = data["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    dt = 1/252
    mu_hat = log_returns.mean() / dt
    sigma_hat = log_returns.std() / np.sqrt(dt)
    plt.figure(figsize=(20,20))
    plt.plot(prices.index, prices["SPY"],linewidth=3)
    plt.ylabel("Price",fontsize=20)
    plt.xlabel("Data",fontsize=20)
    plt.minorticks_on()
    
    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=5, width=1)
    
    plt.grid(which='major', linestyle='-', linewidth=1, color='gray')
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    
    initial_price = prices.loc['2022-01-03 00:00:00']
    return initial_price, mu_hat, sigma_hat

stock_data()