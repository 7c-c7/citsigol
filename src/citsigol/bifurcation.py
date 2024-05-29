import matplotlib.backend_bases
import numpy as np
from matplotlib import pyplot as plt

from citsigol import Map

plt.style.use("dark_background")


class BifurcationDiagram:
    def __init__(
        self,
        map_class: type[Map],
        initial_values: list[float] | None = None,
        steps_to_skip: int = 100,
        n_points: int = 100,
        x_bounds: tuple[float, float] = (0, 1),
        r_bounds: tuple[float, float] = (0, 4),
        resolution: int = 1000,
        max_steps: int = 100_000,
    ):
        self.map_class = map_class
        self.initial_values = initial_values or [0.5]
        self.steps_to_skip = steps_to_skip
        self.n_points = n_points
        self.x_bounds = x_bounds
        self.r_bounds = r_bounds
        self.resolution = resolution
        self.max_steps = max_steps
        self.figure, self.ax = plt.subplots(figsize=(16, 9))
        self.ax.set_ylabel("x")
        self.ax.set_xlabel("r")
        self.r_values = np.linspace(self.r_bounds[0], self.r_bounds[1], self.resolution)
        self.sequences = [
            self.map_class(r).sequence(self.initial_values, self.max_steps)
            for r in self.r_values
        ]

    def _on_lims_change(self, _: matplotlib.backend_bases.Event) -> None:
        for line in self.ax.lines:
            line.remove()
        plt.pause(0.1)
        self.figure.canvas.draw()
        self.x_bounds = self.ax.get_xlim()
        self.r_bounds = self.ax.get_ylim()
        self.r_values = np.linspace(self.r_bounds[0], self.r_bounds[1], self.resolution)
        self.sequences = [
            self.map_class(r).sequence(self.initial_values, self.max_steps)
            for r in self.r_values
        ]
        self.draw()

    def display(self) -> None:
        plt.show(block=False)
        self.reset_axes_limits()
        # self.ax.callbacks.connect("xlim_changed", self._on_lims_change)
        self.ax.callbacks.connect("ylim_changed", self._on_lims_change)
        self.draw()

    def reset_axes_limits(self) -> None:
        self.ax.set_ylim(self.x_bounds)
        self.ax.set_xlim(self.r_bounds)

    def draw(self) -> None:
        for sequence in self.sequences:
            for _ in range(self.steps_to_skip):
                next(sequence)

        plot_pairs = []
        for i in range(self.n_points):
            plot_pairs += [
                (r, x)
                for r, sequence in zip(self.r_values, self.sequences)
                for x in next(sequence)
            ]
            if i % round(self.n_points / 20) == 0 or i == self.n_points - 1:
                rs, xes = zip(*plot_pairs)
                self.ax.plot(
                    rs,
                    xes,
                    ".",
                    markersize=0.3,
                    alpha=0.4,
                    color="aquamarine",
                )
                self.ax.text(
                    0.01,
                    0.99,
                    f"{(i + 1)/self.n_points:.1%}",
                    transform=self.ax.transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=1.0, edgecolor="none"),
                )
                plot_pairs = []
                self.figure.canvas.draw()
                plt.pause(0.00000001)
