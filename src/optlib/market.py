TRADING_DAYS = 252

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Market:
    spot: float           # S0
    rate: float           # r, continuously compounded
    vol: float            # sigma, annualized
    div_yield: float = 0.0  # q (used from Phase 1)


@dataclass(frozen=True)
class Option:
    kind: OptionType
    strike: float         # K
    expiry: float         # T in years
    style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def payoff(self, s):
        if self.kind is OptionType.CALL:
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)


@dataclass
class PricingResult:
    price: float
    std_error: Optional[float] = None   # Monte Carlo only
    greeks: dict = field(default_factory=dict)