import numpy as np


def binomial_tree(S, K, T, r, sigma, n_steps=200, option_type="call",
                  american=True, q=0.0, u=None, d=None):
    """CRR binomial tree for European or American options (Hull Ch 13, Ch 21).

    q is the yield the underlying 'loses' to the holder:
      stock / index -> dividend yield q
      currency      -> foreign risk-free rate r_f
      futures       -> q = r  (growth factor a = 1)
    u, d: optional manual up/down factors (for textbook examples); default CRR.
    """
    dt = T / n_steps
    if u is None:
        u = np.exp(sigma * np.sqrt(dt))
    if d is None:
        d = 1 / u
    a = np.exp((r - q) * dt)              # growth factor
    p = (a - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError(f"p = {p:.4f} outside [0, 1]: no-arbitrage violated, use more steps")
    disc = np.exp(-r * dt)

    def payoff(s):
        return np.maximum(s - K, 0.0) if option_type == "call" else np.maximum(K - s, 0.0)

    j = np.arange(n_steps + 1)
    values = payoff(S * u ** (n_steps - j) * d ** j)
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, payoff(S * u ** (i - j) * d ** j))
    return float(values[0])

def binomial_greeks(S, K, T, r, sigma, n_steps=200, option_type="call",
                    american=True, q=0.0):
    """Price, delta, gamma and theta read off the tree's first nodes (Hull Ch 21).

    Returns per-year theta (divide by TRADING_DAYS for per day)."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    def payoff(s):
        return np.maximum(s - K, 0.0) if option_type == "call" else np.maximum(K - s, 0.0)

    j = np.arange(n_steps + 1)
    values = payoff(S * u ** (n_steps - j) * d ** j)
    saved = {}                                   # option values at time steps 0, 1, 2
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, payoff(S * u ** (i - j) * d ** j))
        if i <= 2:
            saved[i] = values.copy()

    f00 = saved[0][0]
    f11, f10 = saved[1]                          # up node, down node at dt
    f22, f21, f20 = saved[2]                     # uu, ud (= S), dd at 2dt
    delta = (f11 - f10) / (S * u - S * d)
    h = 0.5 * (S * u**2 - S * d**2)
    gamma = ((f22 - f21) / (S * u**2 - S) - (f21 - f20) / (S - S * d**2)) / h
    theta = (f21 - f00) / (2 * dt)
    return {"price": float(f00), "delta": float(delta), "gamma": float(gamma), "theta": float(theta)}