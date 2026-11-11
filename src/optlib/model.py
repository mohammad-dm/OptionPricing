import numpy as np
from scipy.stats import norm


class Stock:
    def __init__(self, stock_price):
        self.stock_price = stock_price

    def generate_path(self, sigma, r, delta_t, n_steps, q=0.0, rng=None):
        rng = rng or np.random.default_rng()
        eps = rng.standard_normal(n_steps)
        log_inc = (r - q - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * eps
        log_path = np.concatenate([[0.0], np.cumsum(log_inc)])
        return self.stock_price * np.exp(log_path)


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """European option price via Black-Scholes."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


# ── Greeks ─────────────────────────────────────────────────────────────────────

def compute_greeks(S, K, T, r, sigma):
    """Return dict of greeks for both call and put."""
    if T <= 0:
        return {g: {"call": 0.0, "put": 0.0} for g in ["delta", "gamma", "theta", "vega", "rho"]}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    phi_d1 = norm.pdf(d1)
    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    Nnd1 = norm.cdf(-d1)
    Nnd2 = norm.cdf(-d2)

    delta_call = Nd1
    delta_put  = Nd1 - 1

    gamma = phi_d1 / (S * sigma * np.sqrt(T))

    theta_call = (
        -S * phi_d1 * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * Nd2
    ) / 365
    theta_put = (
        -S * phi_d1 * sigma / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * Nnd2
    ) / 365

    vega = S * phi_d1 * np.sqrt(T) / 100   # per 1% move in vol

    rho_call =  K * T * np.exp(-r * T) * Nd2  / 100
    rho_put  = -K * T * np.exp(-r * T) * Nnd2 / 100

    return {
        "delta": {"call": delta_call, "put": delta_put},
        "gamma": {"call": gamma,      "put": gamma},
        "theta": {"call": theta_call, "put": theta_put},
        "vega":  {"call": vega,       "put": vega},
        "rho":   {"call": rho_call,   "put": rho_put},
    }


# ── Binomial Tree (American) ──────────────────────────────────────────────────

def binomial_american(S, K, T, r, sigma, n_steps=200, option_type="call"):
    """American option price via CRR binomial tree."""
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal stock prices
    j = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)

    if option_type == "call":
        values = np.maximum(ST - K, 0)
    else:
        values = np.maximum(K - ST, 0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        j = np.arange(i + 1)
        ST_i = S * (u ** (i - j)) * (d ** j)
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if option_type == "call":
            intrinsic = np.maximum(ST_i - K, 0)
        else:
            intrinsic = np.maximum(K - ST_i, 0)
        values = np.maximum(values, intrinsic)

    return float(values[0])


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def monte_carlo(stock, sigma, r, delta_t, n_steps, n_simulations):
    import pandas as pd
    results = []
    for _ in range(n_simulations):
        path = stock.generate_path(sigma, r, delta_t, n_steps)
        results.append(path)
    df = pd.DataFrame(results).T
    df.index.name = "Time Step"
    return df

from optlib.market import Market, Option, ExerciseStyle, PricingResult


def bs_price(option: Option, market: Market) -> PricingResult:
    p = black_scholes(market.spot, option.strike, option.expiry,
                      market.rate, market.vol, option.kind.value)
    return PricingResult(price=p)


def tree_price(option: Option, market: Market, n_steps: int = 200) -> PricingResult:
    if option.style is not ExerciseStyle.AMERICAN:
        raise NotImplementedError("European tree comes in Phase 2")
    p = binomial_american(market.spot, option.strike, option.expiry,
                          market.rate, market.vol, n_steps, option.kind.value)
    return PricingResult(price=p)
