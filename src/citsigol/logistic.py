"""
Logistic map and basic related functions.
"""

from copy import copy
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

CYCLE_REPETITION_ATOL = 1e-8


@dataclass
class LogisticMap:
    """
    Class to represent a logistic map.

    Parameters
    ----------
    r : float
        Parameter of the logistic map.
    """

    r: float

    def iterate_until_convergence(
        self,
        x_0: float,
        max_period: int = 1,
        tol: float = 1e-6,
        max_steps: int = 1_000,
        return_unconverged: bool = False,
    ) -> float | np.ndarray | None:
        """
        Iterate the logistic map until convergence.

        Parameters
        ----------
        x_0 : float
            Initial value of the logistic map.
        max_period : int, optional
            Maximum period to check for convergence, by default 1.
        tol : float, optional
            Tolerance for convergence, by default 1e-6.
        max_steps : int, optional
            Maximum number of iterations, by default 1e6.
        return_unconverged : bool, optional
            Whether to return unconverged values, by default False.

        Returns
        -------
        np.ndarray
            Converged value of the logistic map,
            Or empty array if convergence did not occur and return_unconverged is False
            Or unconverged values of length max_period if return_unconverged is True.
        """
        max_period = min(max_steps, max_period)
        x_history = np.full(max_period, x_0)
        x_new = x_0
        for _ in range(max_steps):
            x_new = logistic_map(x_new, self.r)
            if np.any(np.isclose(x_history, x_new, atol=tol, rtol=0)):
                period_length = max_period - np.max(
                    np.where(np.isclose(x_history, x_new, atol=tol, rtol=0))[0]
                )
                return np.array(x_history[-period_length:])
            x_history = np.roll(x_history, -1)
            x_history[-1] = x_new
        if return_unconverged:
            return x_history
        return np.array([])

    def __call__(self, x: float | np.ndarray, n_steps: int = 1) -> float | np.ndarray:
        """
        Core function of the logistic map.

        Parameters
        ----------
        x : float | np.ndarray
            Current value (or array of values) of the logistic map.
        n_steps : int, optional
            Number of iterations to perform, default 1.

        Returns
        -------
        float | np.ndarray
            Next value of the logistic map (for each value in the array).
        """
        x_n = copy(x)
        for _ in range(n_steps):
            x_n = logistic_map(x_n, self.r)
        return x_n

    def sequence(self, x_0: float, n: int) -> np.ndarray:
        """
        Iterate the logistic map n times.

        Parameters
        ----------
        x_0 : float
            Initial value of the logistic map.
        n : int
            Number of iterations to return (counting the initial value).

        Returns
        -------
        np.ndarray
            Values of the logistic map over n iterations, including the starting value. (length is n+1)
        """
        x_n = np.full(n, np.nan)
        x_n[0] = x_0
        for i in range(1, n):
            x_n[i] = logistic_map(x_n[i - 1], self.r)
        return x_n

    def fixed_points(self, period: int = 1) -> np.ndarray:
        """
        Find the fixed points of the logistic map.
        Choosing a period > 1 will yield limit cycles of length <= period.

        Parameters
        ----------
        period : int
            Maximum number of cycles to

        Returns
        -------
        np.ndarray
            Limit cycle of the logistic map, lowest value first.
        """
        return fixed_points(self.r, period)


def logistic_map(x: float | np.ndarray, r: float) -> float | np.ndarray:
    """
    Logistic map function.

    Parameters
    ----------
    x : float | np.ndarray
        Current value (or array of values) of the logistic map.
    r : float
        Parameter of the logistic map.

    Returns
    -------
    float | np.ndarray
        Next value of the logistic map (for each value in the array).
    """
    return r * x * (1 - x)


def bifurcation_diagram(
    r_values: list[float] | np.ndarray,
    x_0: float = 0.5,
    tol: float = 1e-6,
    n_skip: int = 100,
    n_steps: int = 100,
    track_progress: bool = False,
) -> list[np.ndarray]:
    """
    Generate a bifurcation diagram for the logistic map.

    Parameters
    ----------
    r_values: np.ndarray
        Array of r values to build over.
    x_0 : Vectorizable
        Initial value of the logistic map.
    tol : float, optional
        Tolerance for convergence, by default 1e-6.
    n_skip : int, optional
        Number of iterations to skip before recording values, by default 100.
    n_steps : int, optional
        Number of iterations to record, by default 100.
    track_progress : bool, optional

    Returns
    -------
    list[np.ndarray]
        x values of the bifurcation diagram of the logistic map at each given r value.
    """

    def _make_array(x: float | np.ndarray) -> np.ndarray:
        if isinstance(x, float | int):
            return np.array([x])
        return x

    x_values = []
    max_r = max(r_values)
    print("Iterating...")
    for i, r in enumerate(r_values):
        if track_progress:
            print(f"\rr = {r:.4f}/{max_r}", end="")
        x_values.append(
            _make_array(
                LogisticMap(r).iterate_until_convergence(
                    x_0,
                    n_steps,
                    max_steps=n_skip + n_steps,
                    return_unconverged=True,
                    tol=tol,
                )
            )
        )
    return x_values


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
    if period < 1:
        raise ValueError("Period must be at least 1")
    cycle_polynomial = fixed_point_polynomial(r, period)
    cycle_points = sorted(
        list(
            {
                getattr(root, "real", root)
                for root in cycle_polynomial.roots()
                if not np.iscomplex(root) and 0 <= root <= 1
            }
        )
    )
    logistic = LogisticMap(r)
    cycles = [
        logistic.sequence(point, period) for point in cycle_points
    ]  # cycles of length period starting at each root
    unique_cycles: list[np.ndarray] = []
    for i, cycle in enumerate(cycles):
        # get the indices where the first value recurs in this cycle
        repeats = np.where(np.isclose(cycle, cycle[0], atol=CYCLE_REPETITION_ATOL))[0][
            1:
        ]  # ignore the first
        cycle_length = repeats[0] if repeats.size else cycle.size
        if not any(
            any(np.isclose(unique_cycle, cycle[0])) for unique_cycle in unique_cycles
        ):
            unique_cycles.append(cycle[:cycle_length])
    for cycle in unique_cycles:
        if abs(iterated_logistic_polynomial(r, len(cycle)).deriv()(cycle[0])) <= 1:
            return cycle
    return np.array([])
