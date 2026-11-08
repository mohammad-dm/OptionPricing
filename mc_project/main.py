import os
os.chdir(r"C://Users//m_del//Desktop//Self Study//2026-02-27 Stock price//stock")
from model import Stock
from simulation import monte_carlo
from visualization import plot_histogram, plot_paths
from price_data import stock_data

initial_price, r, sigma = stock_data()
stock = Stock(451.875)

df = monte_carlo(
    stock,
    sigma=0.216415,
    r=0.0877598,
    delta_t=1/365,
    n_steps=365,
    n_simulations=1000
)

plot_histogram(df)
plot_paths(df)