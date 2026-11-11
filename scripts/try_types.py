from optlib.market import Market, Option, OptionType, ExerciseStyle
from optlib.model import bs_price, tree_price

m = Market(spot=42, rate=0.10, vol=0.20)
call = Option(OptionType.CALL, strike=40, expiry=0.5)
put = Option(OptionType.PUT, strike=40, expiry=0.5)
print(bs_price(call, m).price)   # expect 4.76
print(bs_price(put, m).price)    # expect 0.81

am_put = Option(OptionType.PUT, strike=52, expiry=2, style=ExerciseStyle.AMERICAN)
print(tree_price(am_put, Market(spot=50, rate=0.05, vol=0.30), n_steps=2).price)  # expect 7.43 (Figure 13.10)