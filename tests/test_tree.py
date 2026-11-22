import pytest

from optlib.model import black_scholes
from optlib.tree import binomial_tree


def test_two_step_call_fig_13_4():
    c = binomial_tree(20, 21, 0.5, 0.04, None, 2, "call", american=False, u=1.1, d=0.9)
    assert c == pytest.approx(0.9497, abs=0.0005)


def test_two_step_puts_figs_13_7_13_8():
    eu = binomial_tree(50, 52, 2, 0.05, None, 2, "put", american=False, u=1.2, d=0.8)
    am = binomial_tree(50, 52, 2, 0.05, None, 2, "put", american=True, u=1.2, d=0.8)
    assert eu == pytest.approx(4.1927, abs=0.0005)
    assert am == pytest.approx(5.0896, abs=0.0005)


@pytest.mark.parametrize("n, expected", [(2, 7.43), (5, 7.67), (500, 7.47)])
def test_american_put_fig_13_10(n, expected):
    assert binomial_tree(50, 52, 2, 0.05, 0.30, n, "put") == pytest.approx(expected, abs=0.005)


def test_index_call_example_13_1():
    c = binomial_tree(810, 800, 0.5, 0.05, 0.20, 2, "call", american=False, q=0.02)
    assert c == pytest.approx(53.39, abs=0.005)


def test_currency_call_example_13_2():
    c = binomial_tree(0.61, 0.60, 0.25, 0.05, 0.12, 3, "call", american=True, q=0.07)
    assert c == pytest.approx(0.019, abs=0.0005)


def test_futures_put_example_13_3():
    p = binomial_tree(31, 30, 0.75, 0.05, 0.30, 3, "put", american=True, q=0.05)  # futures: q = r
    assert p == pytest.approx(2.84, abs=0.005)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_european_tree_converges_to_bs(kind, q):
    tree = binomial_tree(50, 52, 2, 0.05, 0.30, 2000, kind, american=False, q=q)
    bs = black_scholes(50, 52, 2, 0.05, 0.30, kind, q)
    assert tree == pytest.approx(bs, abs=0.01)


def test_american_call_no_dividends_equals_european():
    am = binomial_tree(50, 52, 2, 0.05, 0.30, 500, "call", american=True)
    eu = binomial_tree(50, 52, 2, 0.05, 0.30, 500, "call", american=False)
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_never_below_european():
    for kind in ["call", "put"]:
        am = binomial_tree(50, 52, 2, 0.05, 0.30, 200, kind, american=True, q=0.06)
        eu = binomial_tree(50, 52, 2, 0.05, 0.30, 200, kind, american=False, q=0.06)
        assert am >= eu - 1e-12