"""
Logistic map and basic related functions.
"""

import numpy as np
from numpy.polynomial import Polynomial

from citsigol import Map


class LogisticMap(Map):
    """
    Class to represent a logistic map.

    Parameters
    ----------
    r : float
        Parameter of the logistic map.
    """

    def __init__(self, r: float):
        self.r = r

        def _logistic_function(x: list[float]) -> list[float]:
            return [r * value * (1 - value) for value in x]

        super().__init__(_logistic_function)


def iterated_logistic_polynomial(r: float, n: int) -> Polynomial:
    """
    Return the nth iterate of the logistic map as a polynomial.

    Parameters
    ----------
    r : float
        Parameter of the logistic map.
    n : int
        Number of iterations.

    Returns
    -------
    Polynomial
        Polynomial representation of the nth iterate of the logistic map.
    """
    x = Polynomial([0, 1])
    for _ in range(n):
        x = r * x * (1 - x)
    return x


def fixed_point_polynomial(r: float, n: int) -> Polynomial:
    """
    Return the Polynomial whose roots are the n-cycles of the logistic map.

    Parameters
    ----------
    r : float
        Parameter of the logistic map.
    n : int
        Number of iterations.

    Returns
    -------
    Polynomial
        Polynomial representation of the nth iterate of the logistic map.
    """
    return Polynomial([0, 1]) - iterated_logistic_polynomial(r, n)


def fixed_points(r: float, period: int = 1) -> np.ndarray:
    """
    Find the fixed points of the logistic map for a given r and initial value.
    Choosing a period > 1 will yield limit cycles of length <= period.

    Parameters
    ----------
    r : float
        Parameter of the logistic map.
    period : int
        Maximum number of cycles to

    Returns
    -------
    np.ndarray
        Limit cycle of the logistic map, lowest value first.
    """
    raise NotImplementedError("This function is not yet implemented.")
