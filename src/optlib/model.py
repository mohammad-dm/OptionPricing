import numpy as np
from scipy.stats import norm
from optlib.market import TRADING_DAYS

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

def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price via Black-Scholes-Merton with dividend yield q."""
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    Sq = S * np.exp(-q * T)        # spot less PV of dividends
    Kr = K * np.exp(-r * T)        # PV of strike
    if option_type == "call":
        return Sq * norm.cdf(d1) - Kr * norm.cdf(d2)
    return Kr * norm.cdf(-d2) - Sq * norm.cdf(-d1)


def compute_greeks(S, K, T, r, sigma, q=0.0):
    """Greeks for call and put (Hull Ch 19, with q).
    theta: per trading day; vega and rho: per 1% move."""
    if T <= 0:
        dc = 1.0 if S > K else 0.0
        dp = -1.0 if S < K else 0.0
        zero = {"call": 0.0, "put": 0.0}
        return {"delta": {"call": dc, "put": dp},
                "gamma": dict(zero), "theta": dict(zero),
                "vega": dict(zero), "rho": dict(zero)}

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf1 = norm.pdf(d1)

    delta_call = eq * norm.cdf(d1)
    delta_put = eq * (norm.cdf(d1) - 1)
    gamma = eq * pdf1 / (S * sigma * np.sqrt(T))

    decay = -S * eq * pdf1 * sigma / (2 * np.sqrt(T))
    theta_call = decay + q * S * eq * norm.cdf(d1) - r * K * er * norm.cdf(d2)
    theta_put = decay - q * S * eq * norm.cdf(-d1) + r * K * er * norm.cdf(-d2)

    vega = S * eq * pdf1 * np.sqrt(T)
    rho_call = K * T * er * norm.cdf(d2)
    rho_put = -K * T * er * norm.cdf(-d2)

    return {
        "delta": {"call": delta_call, "put": delta_put},
        "gamma": {"call": gamma, "put": gamma},
        "theta": {"call": theta_call / TRADING_DAYS, "put": theta_put / TRADING_DAYS},
        "vega":  {"call": vega / 100, "put": vega / 100},
        "rho":   {"call": rho_call / 100, "put": rho_put / 100},
    }


# ── Binomial Tree (American) ──────────────────────────────────────────────────

from optlib.tree import binomial_tree

def binomial_american(S, K, T, r, sigma, n_steps=200, option_type="call", q=0.0):
    return binomial_tree(S, K, T, r, sigma, n_steps, option_type, american=True, q=q)
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
    p = binomial_tree(market.spot, option.strike, option.expiry,
                      market.rate, market.vol, n_steps, option.kind.value,
                      american=(option.style is ExerciseStyle.AMERICAN),
                      q=market.div_yield)
    return PricingResult(price=p)

def _bs_vega_raw(S, K, T, r, sigma, q):
    """dPrice/dsigma per unit of sigma (not per 1%)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                tol=1e-8, max_iter=100):
    """Black-Scholes implied volatility: Newton-Raphson with a bisection safety net."""
    Sq, Kr = S * np.exp(-q * T), K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(Sq - Kr, 0.0), Sq
    else:
        lower, upper = max(Kr - Sq, 0.0), Kr
    if not (lower < price < upper):
        raise ValueError(f"price {price:.4f} outside no-arbitrage bounds "
                         f"({lower:.4f}, {upper:.4f})")

    lo, hi = 1e-6, 5.0              # bracket: sigma between 0.0001% and 500%
    sigma = 0.2                     # starting guess
    for _ in range(max_iter):
        diff = black_scholes(S, K, T, r, sigma, option_type, q) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:                # price too high -> sigma too high
            hi = sigma
        else:
            lo = sigma
        vega = _bs_vega_raw(S, K, T, r, sigma, q)
        new = sigma - diff / vega if vega > 1e-12 else None
        if new is None or not (lo < new < hi):
            new = 0.5 * (lo + hi)   # Newton jumped out of the bracket: bisect
        sigma = new
    raise RuntimeError("implied_vol did not converge")
