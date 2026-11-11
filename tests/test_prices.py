import numpy as np
import pytest

from optlib.market import Market, Option, OptionType, ExerciseStyle
from optlib.model import bs_price, tree_price

M = Market(spot=42, rate=0.10, vol=0.20)          # Example 15.6
CALL = Option(OptionType.CALL, strike=40, expiry=0.5)
PUT = Option(OptionType.PUT, strike=40, expiry=0.5)


def test_bs_call_example_15_6():
    assert bs_price(CALL, M).price == pytest.approx(4.76, abs=0.005)


def test_bs_put_example_15_6():
    assert bs_price(PUT, M).price == pytest.approx(0.81, abs=0.005)


def test_put_call_parity():
    # c - p = S0 - K e^{-rT}  (Ch 11)
    lhs = bs_price(CALL, M).price - bs_price(PUT, M).price
    rhs = M.spot - CALL.strike * np.exp(-M.rate * CALL.expiry)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_american_put_figure_13_10():
    am_put = Option(OptionType.PUT, strike=52, expiry=2, style=ExerciseStyle.AMERICAN)
    p = tree_price(am_put, Market(spot=50, rate=0.05, vol=0.30), n_steps=2).price
    assert p == pytest.approx(7.43, abs=0.005)