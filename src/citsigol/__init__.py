"""Top-level package for citsigol."""

__author__ = """Dustin Phillip Summy"""
__email__ = "dustinsummy@gmail.com"
__version__ = "0.1.0"

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


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


@dataclass
class Citsigol:
    """
    A citsigol (reverse logistic) map.
    """

    r: float

    def __call__(
        self,
        x: Iterable[float] | float,
        branch: int | Iterable[int] = 0,
    ) -> list[float]:
        """
        Evaluate the citsigol map at a point.

        Parameters
        ----------
        x : list[float] | float
            Point(s) at which to evaluate the citsigol map.
        branch : int or Iterable, optional
            Branch of the citsigol map to keep
            Positive value will keep the higher branch
            Negative value will keep the lower branch
            None (default) or 0 will return both values (doubling the size of given x, careful!).

        Returns
        -------
        list[float]
            Value(s) of the citsigol map at x.
        """

        def _branch_indices(chosen_branch: int) -> list[int]:
            return [round((chosen_branch + 1) / 2)] if chosen_branch else [0, 1]

        xes = x if isinstance(x, Iterable) else [x]
        discriminants = [1 - 4 * x_val / self.r for x_val in xes]
        x_priorses = [
            (0.5 * (1 - discriminant**0.5), 0.5 * (1 + discriminant**0.5))
            if discriminant >= 0
            else tuple([])
            for discriminant in discriminants
        ]
        branches = (
            branch if isinstance(branch, Iterable) else len(x_priorses) * [branch]
        )
        if not all(branch in (-1, 0, 1, None) for branch in branches):
            raise ValueError("branch must be -1, 0, 1, or None")
        return [
            x_val
            for branch, possible_xes in zip(branches, x_priorses)
            for i, x_val in enumerate(possible_xes)
            if branch in (0, None) or i in _branch_indices(branch)
        ]

    def sequence(
        self, x_0: float, n: int, compass: Compass | None = None
    ) -> list[list[float]]:
        """
        Iterate the citsigol map n times.

        Parameters
        ----------
        x_0 : float
            Initial value of the citsigol map.
        n : int
            Number of iterations.
        compass : Compass
            Branch of the citsigol map to keep, given the current value of x and the number of steps passed.
            Positive return value will keep the higher branch.
            Negative return value will keep the lower branch.
            if None, keep both branches

        Returns
        -------
        list[float]
            Values of the citsigol map over n iterations, including the starting value. (length is n+1)
        """
        sequence = [[x_0]]
        for i in range(n):
            if x_n := self(sequence[-1], compass(sequence[-1], i) if compass else 0):
                sequence.append(x_n)
                continue
            break
        return sequence


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
