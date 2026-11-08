import numpy as np
class Stock:
    def __init__(self, stock_price):
        self.stock_price = stock_price

    def generate_path(self, sigma, r, delta_t, n_steps):
        eps = np.random.normal(0, 1, n_steps)

        increments = r * delta_t + sigma * np.sqrt(delta_t) * eps
        prices = self.stock_price * np.cumprod(1 + increments)
        prices = np.insert(prices, 0, self.stock_price)

        return prices