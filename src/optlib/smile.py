import pandas as pd

from optlib.model import implied_vol


def smile(chain: pd.DataFrame, spot: float, T: float, r: float, q: float = 0.0) -> pd.DataFrame:
    """Implied vol per strike from out-of-the-money options (puts below spot, calls above)."""
    otm = chain[((chain["kind"] == "call") & (chain["strike"] >= spot))
                | ((chain["kind"] == "put") & (chain["strike"] < spot))]
    rows = []
    for opt in otm.itertuples():
        try:
            iv = implied_vol(opt.mid, spot, opt.strike, T, r, opt.kind, q)
        except (ValueError, RuntimeError):
            continue                     # stale or arbitrageable quote: skip it
        rows.append({"strike": opt.strike, "iv": iv,
                     "yahoo_iv": opt.impliedVolatility, "kind": opt.kind})
    return pd.DataFrame(rows).sort_values("strike")