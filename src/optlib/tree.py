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