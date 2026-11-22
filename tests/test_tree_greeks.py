import pytest

from optlib.market import TRADING_DAYS
from optlib.model import compute_greeks
from optlib.tree import binomial_greeks


def test_hull_american_put_greeks():
    # Ch 21 example: S0 50, K 50, r 10%, sigma 40%, T 5 months, 5 steps
    g = binomial_greeks(50, 50, 5 / 12, 0.10, 0.40, 5, "put")
    assert g["price"] == pytest.approx(4.49, abs=0.005)
    assert g["delta"] == pytest.approx(-0.41, abs=0.005)
    assert g["gamma"] == pytest.approx(0.03, abs=0.005)
    assert g["theta"] == pytest.approx(-4.30, abs=0.005)       # per year


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_greeks_match_formulas(kind):
    g = binomial_greeks(49, 50, 20 / 52, 0.05, 0.20, 1000, kind, american=False)
    bs = compute_greeks(49, 50, 20 / 52, 0.05, 0.20)
    assert g["delta"] == pytest.approx(bs["delta"][kind], abs=1e-3)
    assert g["gamma"] == pytest.approx(bs["gamma"][kind], abs=1e-3)
    assert g["theta"] == pytest.approx(bs["theta"][kind] * TRADING_DAYS, abs=0.01)