"""Top-level package for citsigol."""

__author__ = """Dustin Phillip Summy"""
__email__ = "dustinsummy@gmail.com"
__version__ = "0.1.0"

from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable

import numpy as np


class Map:
    def __init__(self, function: Callable[[list[float]], list[float]]):
        def next_value(x: list[float]) -> list[float]:
            return function(x)

        self._next_value = next_value

    def __call__(self, x: list[float]) -> list[float]:
        return self._next_value(x)

    def sequence(self, x_0: list[float], n: int) -> Generator[list[float], None, None]:
        """
        Iterate the Map at most n times.

        Parameters
        ----------
        x_0 : list[float]
            Initial value of the citsigol map.
        n : int
            Number of iterations.

        Yields
        -------
        list[float]
            Values of the Map after each of the first n iterations, starting with x_0 (length is n+1).
        """
        x_n = [x for x in x_0]
        for _ in range(n):
            if x_n := self(x_n):
                yield x_n
                continue
            break

    def iterate_until_convergence(
        self,
        x_0: float,
        max_period: int = 1,
        tol: float = 1e-6,
        max_steps: int = 1_000,
        skip_steps: int = 0,
        return_unconverged: bool = False,
    ) -> np.ndarray | None:
        """
        Iterate the Map until convergence.

        This method iterates the Map starting from an initial value `x_0` until it converges.
        Convergence is determined by checking if any value in the history of `x` is close to the new `x` within a
        tolerance `tol`. The history of `x` is a rolling window of the last `max_period` values of `x`.
        The method stops if it reaches `max_steps` iterations without finding a convergent value.

        Parameters
        ----------
        x_0 : float
            The initial value of the Map.
        max_period : int, optional
            The maximum period to check for convergence, by default 1.
            This is the size of the rolling window of `x` history.
        tol : float, optional
            The tolerance for convergence, by default 1e-6.
            If any value in the `x` history is within `tol` of the new `x`, the method returns the convergent value.
        max_steps : int, optional
            The maximum number of iterations, by default 1e6.
            If the method reaches this number of iterations without finding a convergent value, it stops and
                returns according to `return_unconverged`.
        skip_steps : int, optional
            The number of iterations to skip before starting to check for convergence, by default 0.
        return_unconverged : bool, optional
            Whether to return unconverged values, by default False.
            If True, the method returns the `x` history when it stops without finding a convergent value.
            If False, the method returns an empty array in this case.

        Returns
        -------
        np.ndarray | None
            If the method finds a convergent value, it returns this value.
            If the method stops without finding a convergent value and `return_unconverged` is True,
                it returns the `x` history.
            If the method stops without finding a convergent value and `return_unconverged` is False,
                it returns an empty array.
        """
        max_period = min(max_steps - skip_steps, max_period)
        x_new = next(
            x
            for i, x in enumerate(self.sequence([x_0], skip_steps), 1)
            if i == skip_steps
        )
        x_history = [x_new]
        for _ in range(max_steps - skip_steps):
            x_new = self(x_new)
            if np.any(
                indices := np.where(np.isclose(x_history, x_new, atol=tol, rtol=0))[0]
            ):
                period_length = len(x_history) - np.max(indices)
                return np.array(x_history[-period_length:])
            if len(x_history) >= max_period:
                x_history = np.roll(x_history, -1)
                x_history[-1] = x_new
            else:
                x_history.append(x_new)
        if return_unconverged:
            return x_history
        return np.array([])


class CitsigolMap(Map):
    """
    A citsigol (reverse logistic) map.
    """

    def __init__(self, r: float):
        self.r = r

        def _citsigol_scalar(x: float) -> list[float]:
            if (discriminant := 1 - 4 * x / self.r) >= 0:
                return [0.5 * (1 - discriminant**0.5), 0.5 * (1 + discriminant**0.5)]
            return []

        def _citsigol_vector(x: list[float]) -> list[float]:
            return [next_val for x_vals in x for next_val in _citsigol_scalar(x_vals)]

        super().__init__(_citsigol_vector)


class Compass(ABC):
    """
    A compass for the citsigol map.

    A callable function that accepts the current value of x and the number of steps passed, and returns +1 or -1 to
    determine which branch of the citsigol map to follow.
    """

    @abstractmethod
    def __call__(self, x: float | list[float], n: int) -> list[int]:
        """
        Choose a branch of the citsigol map to follow.

        Parameters
        ----------
        x : float | list[float]
            Current value of x.
        n : int
            Number of steps passed.

        Returns
        -------
        int
            +1 to follow the higher branch, -1 to follow the lower branch.
        """
        raise NotImplementedError("Compass must be implemented as a callable function.")


class Seeker(Compass):
    """
    A seeker for the citsigol map.

    A callable function that accepts the current value of x and the number of steps passed, and returns +1 or -1 to
    determine which branch of the citsigol map to follow.
    """

    def __init__(self, target: float):
        self.target = target

    def __call__(self, x: float | list[float], n: int) -> list[int]:
        """
        Choose a branch of the citsigol map to follow.

        Parameters
        ----------
        x : float
            Current value of x.
        n : int
            Number of steps passed.

        Returns
        -------
        int
            +1 to follow the higher branch, -1 to follow the lower branch.
        """
        return [
            int(np.sign(self.target - x_value) or 1)
            for x_value in (x if isinstance(x, list) else [x])
        ]  # default to higher branch if x == target


class Quest(Compass):
    """
    A map quest (list of fixed directions to follow in order) for the citsigol map.
    """

    def __init__(self, directions: list[int] | Callable[[int], int]):
        self.directions = (
            directions.__getitem__ if isinstance(directions, list) else directions
        )

    def __call__(self, x: float | list[float], n: int) -> list[int]:
        """
        Choose a branch of the citsigol map to follow.

        Parameters
        ----------
        x : float
            Current value of x.
        n : int
            Number of steps passed.

        Returns
        -------
        int
            +1 to follow the higher branch, -1 to follow the lower branch.
        """
        return (len(x) if isinstance(x, list) else 1) * [self.directions(n)]


def citsigol_branch(
    x: Iterable[float] | float, r: float, branch: int | Iterable[int]
) -> list[float]:
    """
    Evaluate the citsigol map at a point.

    Parameters
    ----------
    x : list[float] | float
        Point(s) at which to evaluate the citsigol map.
    r : float
        Parameter of the citsigol map.
    branch : int | Iterable[int]
        Branch of the citsigol map to keep
        Positive value will keep the higher branch
        Negative value (or zero) will keep the lower branch

    Returns
    -------
    list[float]
        Value(s) of the citsigol map at x.
    """

    xes = x if isinstance(x, list) else [x]
    branch_choices = branch if isinstance(branch, Iterable) else len(xes) * [branch]

    def _sign(val: float) -> int:
        return int(np.sign(val)) or -1

    discriminants = [1 - 4 * x_val / r for x_val in xes]
    return [
        0.5 * (1 + _sign(branch_choice) * discriminant**0.5)
        for discriminant, branch_choice in zip(discriminants, branch_choices)
        if discriminant >= 0
    ]
