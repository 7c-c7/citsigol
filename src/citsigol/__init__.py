"""Top-level package for citsigol."""

__author__ = """Dustin Phillip Summy"""
__email__ = "dustinsummy@gmail.com"
__version__ = "0.1.0"

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


class Compass(ABC):
    """
    A compass for the citsigol map.

    A callable function that accepts the current value of x and the number of steps passed, and returns +1 or -1 to
    determine which branch of the citsigol map to follow.
    """

    @abstractmethod
    def __call__(self, x: float, n: int) -> int:
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
        raise NotImplementedError("Compass must be implemented as a callable function.")


@dataclass
class Citsigol:
    """
    A citsigol (reverse logistic) map.
    """

    r: float

    def __call__(
        self,
        x: np.ndarray,
        branch: int = 0,
    ) -> np.ndarray:
        """
        Evaluate the citsigol map at a point.

        Parameters
        ----------
        x : float or np.ndarray
            Point(s) at which to evaluate the citsigol map.
        branch : int, float, np.ndarray, or Callable, optional
            Branch of the citsigol map to keep
            Positive value will keep the higher branch
            Negative value will keep the lower branch
            None (default) or 0 will return both values (doubling the size of given x, careful!).

        Returns
        -------
        np.ndarray
            Value(s) of the citsigol map at x.
        """
        if branch not in (-1, 0, 1, None):
            raise ValueError("branch must be -1, 0, 1, or None")
        if len(x.shape) != 1:
            raise ValueError("x must be a 1D array")
        d_squared = 1 - 4 * x / self.r
        if d_squared < 0:
            return np.array([])
        distance = np.sqrt(d_squared)
        branches = [0.5 * (1 - distance), 0.5 * (1 + distance)]

        def _index(branch_choice: int) -> int:
            return round((1 + branch_choice) / 2)

        return np.concatenate(
            np.array([branches[_index(branch)]])
            if branch in (-1, 1)
            else np.array([branches[0], branches[1]])
        )

    def sequence(self, x_0: float, n: int, compass: Compass) -> list[float]:
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

        Returns
        -------
        list[float]
            Values of the citsigol map over n iterations, including the starting value. (length is n+1)
        """
        sequence = [x_0]
        for i in range(n):
            if np.size(x_n := self(np.array(sequence[-1:]), compass(sequence[-1], i))):
                sequence.extend(x_n)
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

    def __call__(self, x: float, n: int) -> int:
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
        return np.sign(self.target - x) or 1  # default to higher branch if x == target


class Quest(Compass):
    """
    A map quest (list of fixed directions to follow in order) for the citsigol map.
    """

    def __init__(self, directions: list[int] | Callable[[int], int]):
        self.directions = (
            directions.__getitem__ if isinstance(directions, list) else directions
        )

    def __call__(self, x: float, n: int) -> int:
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
        return self.directions(n)
