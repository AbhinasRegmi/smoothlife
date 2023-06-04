from math import exp
from dataclasses import dataclass


@dataclass(frozen=True)
class MagicNums:
    # outer_radius: float = 11.0
    outer_radius: float = 50.0
    alpha: float = 0.028
    b1: float = 0.278
    b2: float = 0.365
    d1: float = 0.267
    d2: float = 0.445
    dt: float = 0.05


class MagicFunc:
    @staticmethod
    def _sigma_1(x: float, a: float) -> float:
        return 1.0 / ( 1.0 + exp( - ( x - a) * 4 / MagicNums.alpha ) )
    
    @staticmethod
    def _sigma_2(x: float, a: float, b: float) -> float:
        return MagicFunc._sigma_1(x, a) * ( 1 - MagicFunc._sigma_1(x, b))
    
    @staticmethod
    def _sigma_m(x: float, y: float, m: float) -> float:
        return x * (1 - MagicFunc._sigma_1(m, 0.5)) + y * MagicFunc._sigma_1(m, 0.5)
    
    @staticmethod
    def next_state(n: float, m: float) -> float:
        return MagicFunc._sigma_2(
            n,
            MagicFunc._sigma_m(
                MagicNums.b1,
                MagicNums.d1,
                m
            ),
            MagicFunc._sigma_m(
                MagicNums.b2,
                MagicNums.d2,
                m
            )
        )