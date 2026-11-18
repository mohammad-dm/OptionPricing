import numpy as np
import pytest

from optlib.market import TRADING_DAYS
from optlib.model import black_scholes, compute_greeks


def test_index_call_with_q():
    # Ch 17 index-option example: S0 930, K 900, r 8%, q 3%, sigma 20%, T = 2 months
    c = black_scholes(930, 900, 2 / 12, 0.08, 0.20, "call", q=0.03)
    assert c == pytest.approx(51.83, abs=0.005)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity_with_q(q):
    S, K, T, r, s = 50, 55, 0.75, 0.04, 0.25
    c = black_scholes(S, K, T, r, s, "call", q)
    p = black_scholes(S, K, T, r, s, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_greeks_ch19_example():
    g = compute_greeks(49, 50, 20 / 52, 0.05, 0.20)
    assert g["delta"]["call"] == pytest.approx(0.522, abs=0.0005)
    assert g["gamma"]["call"] == pytest.approx(0.066, abs=0.0005)
    assert g["theta"]["call"] * TRADING_DAYS == pytest.approx(-4.31, abs=0.005)  # per year
    assert g["vega"]["call"] * 100 == pytest.approx(12.1, abs=0.05)              # per unit vol
    assert g["rho"]["call"] * 100 == pytest.approx(8.91, abs=0.005)              # per unit r


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, s, q = 100, 95, 0.5, 0.03, 0.25, 0.02
    base = dict(S=S, K=K, T=T, r=r, sigma=s, q=q)
    f = lambda **bump: black_scholes(**{**base, **bump}, option_type=kind)
    g = compute_greeks(S, K, T, r, s, q)
    h = 1e-4
    delta = (f(S=S + h) - f(S=S - h)) / (2 * h)
    gamma = (f(S=S + h) - 2 * f() + f(S=S - h)) / h**2
    vega = (f(sigma=s + h) - f(sigma=s - h)) / (2 * h)
    rho = (f(r=r + h) - f(r=r - h)) / (2 * h)
    theta = -(f(T=T + h) - f(T=T - h)) / (2 * h)   # time passing = T shrinking
    assert g["delta"][kind] == pytest.approx(delta, abs=1e-6)
    assert g["gamma"][kind] == pytest.approx(gamma, abs=1e-4)
    assert g["vega"][kind] * 100 == pytest.approx(vega, abs=1e-5)
    assert g["rho"][kind] * 100 == pytest.approx(rho, abs=1e-5)
    assert g["theta"][kind] * TRADING_DAYS == pytest.approx(theta, abs=1e-5)


def test_expiry_delta():
    assert compute_greeks(110, 100, 0, 0.05, 0.2)["delta"]["call"] == 1.0
    assert compute_greeks(90, 100, 0, 0.05, 0.2)["delta"]["call"] == 0.0
    assert compute_greeks(90, 100, 0, 0.05, 0.2)["delta"]["put"] == -1.0