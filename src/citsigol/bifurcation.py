from __future__ import annotations

import dataclasses
import typing

import matplotlib
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np

import citsigol

FIG_SIZE = (16, 9)
PLOT_DRAW_PAUSE_TIME = (
    1e-8  # pausing is required when drawing the plot to avoid skipping the drawing.
)

plt.style.use("dark_background")


@dataclasses.dataclass
class BifurcationDiagramConfig:
    """
    A dataclass used to represent the configuration of a Bifurcation Diagram.

    ...

    Attributes
    ----------
    parametrized_map : ParametrizedMap
        a ParametrizedMap that represents the map used in the bifurcation diagram, for filling in defaults
    initial_values : list[float]
        a list of initial values for the map sequences (default is [0.5])
    steps_to_skip : int
        the number of steps to skip in each sequence before plotting (default is 100)
    n_points : int
        the number of points to plot in each sequence (default is 100)
    x_bounds : tuple[float, float]
        the bounds for the x-axis (default is (0, 1))
    parameter_bounds : tuple[float, float]
        the bounds for the r-axis (default is (0, 4))
    resolution : int
        the resolution of the r-values (default is 1000)
    max_steps : int
        the maximum number of steps in each sequence (default is 100_000)
    """

    parametrized_map: citsigol.ParametrizedMap | None = None
    initial_values: list[float] = dataclasses.field(default_factory=lambda: [0.5])
    steps_to_skip: int = None  # type: ignore
    n_points: int = None  # type: ignore
    x_bounds: tuple[float, float] = None  # type: ignore
    parameter_bounds: tuple[float, float] = None  # type: ignore
    resolution: int = None  # type: ignore
    max_steps: int = None  # type: ignore

    def __post_init__(self) -> None:
        if self.parametrized_map is not None:
            for field in dataclasses.fields(self):
                if getattr(self, field.name, None) is None:
                    setattr(
                        self, field.name, getattr(self.parametrized_map, field.name)
                    )


class BifurcationDiagram:
    """
    A class used to represent a Bifurcation Diagram.

    Attributes
    ----------
    parametrized_map : ParametrizedMap
        a ParametrizedMap that represents the map used in the bifurcation diagram
    config : BifurcationDiagramConfig
        the configuration for the bifurcation diagram, see BifurcationDiagramConfig for attributes
    figure : matplotlib.figure.Figure
        the figure object of the plot
    ax : matplotlib.axes.Axes
        the axes object of the plot
    r_values : numpy.ndarray
        the r-values used in the plot
    sequences : list[Iterator]
        the sequences of the map

    Methods
    -------
    _on_lims_change(_: matplotlib.backend_bases.Event) -> None
        Handles the event when the limits of the axes are changed.
    display() -> None
        Displays the bifurcation diagram.
    reset_axes_limits() -> None
        Resets the limits of the axes to the original bounds.
    draw() -> None
        Draws the bifurcation diagram.
    """

    def __init__(
        self,
        parametrized_map: citsigol.ParametrizedMap,
        config: BifurcationDiagramConfig,
        **kwargs: typing.Any,
    ):
        """
        Constructs all the necessary attributes for the BifurcationDiagram object.

        Parameters
        ----------
            parametrized_map : ParametrizedMap
                a ParametrizedMap that represents the map used in the bifurcation diagram
            config : BifurcationDiagramConfig
                the configuration for the bifurcation diagram, see BifurcationDiagramConfig for attributes
            kwargs : dict
                additional keyword arguments for the plot, mimicking matplotlib.pyplot.subplots kwargs
        """
        if "fig_size" not in kwargs:
            kwargs["fig_size"] = FIG_SIZE
        self.parametrized_map = parametrized_map
        self.config = config
        for field in dataclasses.fields(self.config):
            setattr(self, field.name, getattr(self.config, field.name))

        self.total_points_to_plot = self.config.n_points * self.config.resolution
        self.zoom_box: list[tuple[float, float]] = [(0, 0), (0, 0)]
        matplotlib.rcParams["toolbar"] = "None"
        self.figure, self.ax = plt.subplots(figsize=kwargs["fig_size"])
        matplotlib.rcParams["toolbar"] = "toolbar2"
        self.ax.set_ylabel("x")
        self.ax.set_xlabel(self.parametrized_map.parameter_name)
        self.r_values = np.linspace(
            self.config.parameter_bounds[0],
            self.config.parameter_bounds[1],
            self.config.resolution,
        )
        self.sequences = self.generator_sequences()
        self.progress_text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=1.0, edgecolor="none"),
        )
        self._proceed = True

    def generator_sequences(self) -> list[typing.Iterator[list[float]]]:
        """
        Generate sequence generator functions of the map for each of self.r_values.

        Returns
        -------
        list[Iterator]
            the sequences of the map
        """
        return [
            self.parametrized_map.map_instance(r).sequence(self.config.initial_values)
            for r in self.r_values
        ]

    def _begin_zoom_box(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """
        Collect starting point of zoom box from mouse press event.

        Parameters
        ----------
            event : matplotlib.backend_bases.Event
                the event object
        """
        self._proceed = False
        self.zoom_box[0] = (event.xdata, event.ydata)

    def _finish_zoom_box(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """
        Collect ending point of zoom box from mouse release event, update axes limits and begin redrawing.

        Parameters
        ----------
            event : matplotlib.backend_bases.Event
                the event object
        """
        self._proceed = True
        self.zoom_box[1] = (event.xdata, event.ydata)
        self.restart()

    def restart(self) -> None:
        """
        Restart the bifurcation diagram. In case of change of bounds, for example.
        """
        self._update_bounds()
        self._clear_points()
        self.r_values = np.linspace(
            self.config.parameter_bounds[0],
            self.config.parameter_bounds[1],
            self.config.resolution,
        )
        self.sequences = self.generator_sequences()
        self.populate()

    def _update_bounds(self) -> None:
        """
        Update the bounds of the axes based on the zoom box.
        """
        x0, y0 = self.zoom_box[0]
        x1, y1 = self.zoom_box[1]
        self.config.parameter_bounds = (min(x0, x1), max(x0, x1))
        self.config.x_bounds = (min(y0, y1), max(y0, y1))
        self.ax.set_xlim(self.config.parameter_bounds)
        self.ax.set_ylim(self.config.x_bounds)

    def _clear_points(self) -> None:
        """
        Remove any datapoints from the figure or axes that are not inside the current x_bounds and parameter_bounds.
        """
        while self.ax.lines:
            self.ax.lines[0].remove()
        self.draw()

    def display(self) -> None:
        """
        display the bifurcation diagram.

        Show the bifurcation diagram plot.
        """
        matplotlib.rcParams["toolbar"] = "None"
        plt.show(block=False)
        matplotlib.rcParams["toolbar"] = "toolbar2"
        self.reset_axes_limits()
        self.figure.canvas.mpl_connect("button_press_event", self._begin_zoom_box)
        self.figure.canvas.mpl_connect("button_release_event", self._finish_zoom_box)
        self.populate()

    def reset_axes_limits(self) -> None:
        """
        Set axes limits to match bifurcation diagram bounds.
        """
        self.ax.set_ylim(self.config.x_bounds)
        self.ax.set_xlim(self.config.parameter_bounds)

    def _skip_initial_data(self) -> None:
        """
        Skip the initial data points in the sequences.
        """
        self.progress_text.set_visible(True)
        self.progress_text.set_text("Skipping initial data...")
        self.draw()
        points_found = np.full(self.config.resolution, 0)
        while self._proceed and np.all(points_found < self.config.steps_to_skip):
            points_found = np.add(
                points_found,
                [
                    self.config.steps_to_skip
                    if not (
                        num_found := len(
                            [
                                True
                                for _ in range(self.config.steps_to_skip)
                                for x in next(sequence, [])
                                if self.config.x_bounds[0]
                                <= x
                                <= self.config.x_bounds[1]
                            ]
                        )
                        if points_found_this_sequence < self.config.steps_to_skip
                        else 0
                    )
                    and np.any(points_found > self.config.steps_to_skip)
                    else num_found
                    for points_found_this_sequence, sequence in zip(
                        points_found, self.sequences
                    )
                ],
            )

    def populate(self) -> None:
        """
        Draw the bifurcation diagram. Updates the plot frame.
        """
        self._skip_initial_data()
        plot_pairs: list[tuple[float, float]] = []
        points_found = np.full(self.config.resolution, 0)
        total_points_found = 0
        chunk_size = 1 / 20
        while self._proceed and total_points_found < self.total_points_to_plot:
            while (
                self._proceed
                and len(plot_pairs) / self.total_points_to_plot <= chunk_size
            ):
                for r, sequence, points_found_this_sequence in zip(
                    self.r_values, self.sequences, points_found
                ):
                    if points_found_this_sequence < self.config.steps_to_skip:
                        plot_pairs += [  # add points to the plot if they are within bounds
                            (r, x)
                            for x in next(sequence, [])
                            if self.config.parameter_bounds[0]
                            <= r
                            <= self.config.parameter_bounds[1]
                            and self.config.x_bounds[0] <= x <= self.config.x_bounds[1]
                        ]
            total_points_found += len(plot_pairs)
            rs, xes = zip(*plot_pairs)
            self.ax.plot(
                rs,
                xes,
                ".",
                markersize=0.3,
                alpha=0.4,
                color="aquamarine",
            )
            if total_points_found < self.total_points_to_plot:
                self.progress_text.set_visible(True)
                self.progress_text.set_text(
                    f"{total_points_found / self.total_points_to_plot:6.1%}"
                )
            else:
                self.progress_text.set_visible(False)
            plot_pairs = []
            self.draw()

    def draw(self) -> None:
        """
        Updates the plot frame.
        """
        self.figure.canvas.draw()
        plt.pause(PLOT_DRAW_PAUSE_TIME)
