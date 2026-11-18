import numpy as np
import pytest

from optlib.model import black_scholes, implied_vol


def test_hull_implied_vol_example():
    # Ch 15: call worth 1.875 with S0 21, K 20, r 10%, T 0.25 -> sigma about 23.5%
    assert implied_vol(1.875, 21, 20, 0.25, 0.10, "call") == pytest.approx(0.235, abs=0.0005)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_round_trip(kind, K, sigma):
    S, T, r, q = 100, 0.5, 0.04, 0.02
    price = black_scholes(S, K, T, r, sigma, kind, q)
    fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
    lower = max(fwd, 0) if kind == "call" else max(-fwd, 0)
    if price - lower < 1e-6:
        pytest.skip("almost no time value: sigma can't be recovered")
    assert implied_vol(price, S, K, T, r, kind, q) == pytest.approx(sigma, abs=1e-6)


def test_rejects_price_below_intrinsic():
    with pytest.raises(ValueError):
        implied_vol(0.5, 42, 40, 0.5, 0.10, "call")   # below S - K e^{-rT} = 3.90


@pytest.mark.parametrize("S", [80, 95, 100, 105, 120])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity_grid(S, T, q):
    K, r, s = 100, 0.05, 0.25
    c = black_scholes(S, K, T, r, s, "call", q)
    p = black_scholes(S, K, T, r, s, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)