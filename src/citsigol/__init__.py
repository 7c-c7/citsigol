"""Top-level package for citsigol."""

__author__ = """Dustin Phillip Summy"""
__email__ = "dustinsummy@gmail.com"
__version__ = "0.1.0"

import dataclasses
import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import citsigol.bifurcation as bf

# Dynamically retrieve the parameters of plt.subplots and ax.plot
PYPLOT_SUBPLOTS_KWARGS = {"figsize", "dpi"}
PYPLOT_LINE_COLLECTION_KWARGS = {
    "edgecolors",
    "facecolors",
    "linestyles",
    "linewidths",
    "alpha",
}


class Map:
    def __init__(self, function: typing.Callable[[list[float]], list[float]]):
        def next_value(x: list[float]) -> list[float]:
            return function(x)

        self._next_value = next_value

    def __call__(self, x: list[float]) -> list[float]:
        return self._next_value(x)

    def sequence(
        self, x_0: list[float], n: int | None = None
    ) -> typing.Generator[list[float], None, None]:
        """
        Iterate the Map at most n times.

        Parameters
        ----------
        x_0 : list[float]
            Initial value of the citsigol map.
        n : int | None
            Number of iterations. If None, iterate indefinitely.

        Yields
        -------
        list[float]
            Values of the Map after each of the first n iterations, starting with x_0 (length is n+1).
        """
        x_n = [x for x in x_0]
        while n is None:
            if x_n := self(x_n):
                yield x_n
                continue
            break
        else:
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

    def plot(
        self,
        x_0: list[float],
        n_steps: int = 100,
        ax: plt.Axes = None,
        **kwargs: typing.Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the Map on the given axes, or create a new figure and axes to plot on.

        Plots the Map starting from the initial value `x_0` for `n_steps` iterations.

        All subsequent points will be connected by lines to their origin in the x vector (lines with multiple branches
        will fork at each iteration).

        Parameters
        ----------
        x_0 : list[float]
            Initial value(s).
        n_steps : int, optional
            Number of iterations to plot, by default 100.
        ax : plt.Axes, optional
            Axes to plot on.
            If None (default), a new figure and axes will be created.
        **kwargs
            Additional keyword arguments to pass to plt.subplots and ax.plot.

        Returns
        -------
        plt.Figure, plt.Axes
            Figure and axes of the plot.
        """

        # Filter kwargs for plt.subplots and ax.plot
        subplots_kwargs = {
            k: v for k, v in kwargs.items() if k in PYPLOT_SUBPLOTS_KWARGS
        }
        plot_kwargs = {
            k: v for k, v in kwargs.items() if k in PYPLOT_LINE_COLLECTION_KWARGS
        }
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = self.__repr__()

        if ax is None:
            fig, ax = plt.subplots(**subplots_kwargs)
        else:
            fig = ax.figure

        lines: list[tuple[tuple[int, float], tuple[int, float]]] = []
        x_vals = x_0.copy()
        for i in range(n_steps):
            next_x = []
            for x_i in x_vals:
                next_x += (result_xs := self([x_i]))
                lines += [((i, x_i), (i + 1, x_i_plus_1)) for x_i_plus_1 in result_xs]
            x_vals = next_x.copy()

        line_collection = LineCollection(lines, **plot_kwargs)
        line_collection.set_label(kwargs["label"])
        ax.add_collection(line_collection)
        ax.autoscale()
        fig.canvas.draw()
        return fig, ax


@dataclasses.dataclass
class ParametrizedMap:
    parametrized_function: typing.Callable[[list[float], float], list[float]]
    parameter_name: str = dataclasses.field(default_factory=lambda: "r")
    initial_values: list[float] | None = None
    steps_to_skip: int = 100
    n_points: int = 100
    x_bounds: tuple[float, float] = (0, 1)
    parameter_bounds: tuple[float, float] = (0, 4)
    resolution: int = 1000
    max_steps: int = 100_000

    def __post_init__(self) -> None:
        self.function = self.parametrized_function

    def __call__(self, x: list[float], *args: float, **kwargs: float) -> list[float]:
        return self.function(x, *args, **kwargs)

    def bifurcation_diagram(
        self,
        config: bf.BifurcationDiagramConfig | None = None,
        **kwargs: typing.Any,
    ) -> bf.BifurcationDiagram:
        """
        Create the bifurcation diagram of the Map on the given axes, or create a new figure and axes to plot on.

        Parameters
        ----------
        config: BifurcationDiagramConfig | None
            Configuration for the bifurcation diagram.
            If None, a default configuration will be used based on the attributes of the ParametrizedMap.

        Returns
        -------
        plt.Axes
            Axes of the plot.
        """
        config = (
            bf.BifurcationDiagramConfig(parametrized_map=self)
            if config is None
            else config
        )
        return bf.BifurcationDiagram(self, config, **kwargs)

    def map_instance(self, *args: float, **kwargs: float) -> Map:
        """
        Build and return an instance of the Map with the parameters of the ParametrizedMap.

        Parameters
        ----------
        *args
            Additional positional arguments to pass to the Map.
        **kwargs
            Additional keyword arguments to pass to the Map.

        Returns
        -------
        Map
            Instance of the Map with the parameters of the ParametrizedMap.
        """

        def map_function(x: list[float]) -> list[float]:
            return self(x, *args, **kwargs)

        return Map(function=map_function)


def _citsigol_scalar(x: float, r: float) -> list[float]:
    if r and (discriminant := 1 - 4 * x / r) >= 0:
        return [0.5 * (1 - discriminant**0.5), 0.5 * (1 + discriminant**0.5)]
    return []


def _citsigol_vector(x: list[float], r: float) -> list[float]:
    return [next_val for x_val in x for next_val in _citsigol_scalar(x_val, r)]


citsigol_map = ParametrizedMap(
    parametrized_function=_citsigol_vector,
    parameter_name="r",
    steps_to_skip=10,
    n_points=100,
    x_bounds=(0, 1),
    parameter_bounds=(0, 4),
    resolution=100,
    max_steps=1000,
)
