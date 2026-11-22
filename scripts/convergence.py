"""Phase 2: how the binomial tree converges to Black-Scholes, and the control variate.

Run from the project folder:  py scripts/convergence.py
Saves convergence.png next to where you run it.
"""
import numpy as np
import matplotlib.pyplot as plt

from optlib.model import black_scholes
from optlib.tree import binomial_tree

S, K, T, r, sigma = 50, 52, 2, 0.05, 0.30          # the Figure 13.10 option

# 1. European put: tree price for n = 1..300 against Black-Scholes
bs = black_scholes(S, K, T, r, sigma, "put")
ns = np.arange(1, 301)
eu = np.array([binomial_tree(S, K, T, r, sigma, n, "put", american=False) for n in ns])
avg = (eu[:-1] + eu[1:]) / 2                         # average of n and n+1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(ns, eu, lw=0.8, label="tree, n steps")
ax1.plot(ns[:-1], avg, lw=1.5, label="average of n and n+1")
ax1.axhline(bs, color="k", ls="--", label=f"Black-Scholes {bs:.4f}")
ax1.set_ylim(bs - 0.15, bs + 0.15)
ax1.set_xlabel("steps n"); ax1.set_ylabel("European put price"); ax1.legend()
ax1.set_title("Zig-zag convergence")

ax2.loglog(ns, np.abs(eu - bs), lw=0.8, label="|tree - BS|")
ax2.loglog(ns[:-1], np.abs(avg - bs), lw=1.5, label="|average - BS|")
ax2.loglog(ns, 1.0 / ns, "k:", label="1/n reference")
ax2.set_xlabel("steps n"); ax2.set_ylabel("absolute error"); ax2.legend()
ax2.set_title("Error shrinks like 1/n")
fig.tight_layout(); fig.savefig("convergence.png", dpi=120)
print("saved convergence.png")

# 2. Control variate for the American put (Hull Ch 21)
ref = binomial_tree(S, K, T, r, sigma, 10_000, "put", american=True)   # "true" value
print(f"\nAmerican put reference (n = 10,000): {ref:.5f}")
print(f"{'n':>5} {'plain tree':>11} {'error':>9} {'control var.':>13} {'error':>9}")
for n in [5, 10, 25, 50, 100, 200]:
    am = binomial_tree(S, K, T, r, sigma, n, "put", american=True)
    eu_n = binomial_tree(S, K, T, r, sigma, n, "put", american=False)
    cv = am + (bs - eu_n)            # correct the American price by the European tree's error
    print(f"{n:>5} {am:>11.5f} {am - ref:>+9.5f} {cv:>13.5f} {cv - ref:>+9.5f}")