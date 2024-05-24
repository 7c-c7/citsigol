"""Top-level package for citsigol."""

__author__ = """Dustin Phillip Summy"""
__email__ = "dustinsummy@gmail.com"
__version__ = "0.1.0"

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CitsigolMap:
    """
    A citsigol (reverse logistic) map.
    """

    r: float

    def __call__(
        self,
        x: float | np.ndarray,
        branch: float | np.ndarray | Callable[[float], int] = None,
    ) -> float | np.ndarray:
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
            None (default) will return both values (doubling the size of given x, careful!).
            If branch_choices is callable, it should return the branch choice for each x.

        Returns
        -------
        float
            Value of the citsigol map at x.
        """
        if isinstance(x, np.ndarray) and len(x.shape) != 1:
            raise ValueError("x must be a 1D array")
        branch_choices = (
            np.array(
                [
                    np.sign(value or 0)
                    for value in (branch if isinstance(branch, Iterable) else [branch])
                ]
            )
            if branch
            else np.full(len(x) if isinstance(x, np.ndarray) else 1, 0.0)
        )
        distance = np.sqrt(1 - 4 * x / self.r)
        branches = [0.5 * (1 - distance), 0.5 * (1 + distance)]

        def _index(branch_choice: int) -> int:
            return round((1 + branch_choice) / 2)

        return_val = np.concatenate(
            [
                branches[_index(branch)]
                if branch in (-1, 1)
                else [branches[0], branches[1]]
                for branch in branch_choices
            ]
        )
        return (
            float(return_val[0])
            if isinstance(x, float | int) and len(return_val) == 1
            else return_val
        )

    def iterate(self, x_0: float, n: int) -> float:
        """
        Iterate the citsigol map n times.

        Parameters
        ----------
        x_0 : float
            Initial value of the citsigol map.
        n : int
            Number of iterations.

        Returns
        -------
        float
            Value of the citsigol map after n iterations.
        """
        x = x_0
        for _ in range(n):
            x = self(x)
        return x
