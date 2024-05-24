"""
Logistic map and basic related functions.
"""

from copy import copy
from dataclasses import dataclass
from numbers import Number

import numpy as np


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
        x_0: float | np.ndarray,
        max_period: int = 1,
        tol: float = 1e-6,
        max_steps: int = 1_000_000,
        return_unconverged: bool = False,
    ) -> float | np.ndarray | None:
        """
        Iterate the logistic map until convergence.

        Parameters
        ----------
        x_0 : float | np.ndarray
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
        float | np.ndarray | None
            Converged value of the logistic map,
            Or None if convergence did not occur and return_unconverged is False
            Or unconverged values of length max_period if return_unconverged is True.
        """
        x_history = np.full(max_period, x_0)
        x = copy(x_0)
        for _ in range(max_steps):
            x_new = self(x)
            if np.any(
                [np.allclose(x_new, x_prev, atol=tol, rtol=0) for x_prev in x_history]
            ):
                period_length = next(
                    i
                    for i, x_prev in enumerate(reversed(x_history))
                    if np.all(np.abs(x_prev - x_new) < tol)
                )
                if period_length == 1 and isinstance(x_0, Number):
                    return float(x_history[-1])
                else:
                    return np.array(x_history[-period_length:])
            x = copy(x_new)
            x_history = np.roll(x_history, -1)
            x_history[-1] = x_new
        if return_unconverged:
            return x_history
        return None

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
    r_values: np.ndarray,
    x_0: float = 0.5,
    n_skip: int = 100,
    n_steps: int = 100,
) -> list[np.ndarray]:
    """
    Generate a bifurcation diagram for the logistic map.

    Parameters
    ----------
    r_values: np.ndarray
        Array of r values to build over.
    x_0 : Vectorizable
        Initial value of the logistic map.
    n_skip : int, optional
        Number of iterations to skip before recording values, by default 100.
    n_steps : int, optional
        Number of iterations to record, by default 100.

    Returns
    -------
    list[np.ndarray]
        x values of the bifurcation diagram of the logistic map at each given r value.
    """
    return [
        LogisticMap(r).iterate_until_convergence(
            x_0, n_steps, max_steps=n_skip + n_steps, return_unconverged=True
        )
        for r in r_values
    ]
