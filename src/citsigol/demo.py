"""Demo script for citsigol."""

import math

import matplotlib.pyplot as plt
import numpy as np

import citsigol
import citsigol.logistic as logistic
from citsigol.bifurcation import BifurcationDiagram, BifurcationDiagramConfig


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")
    print(
        "Plotting examples of the logistic map and the citsigol map over several iterations..."
    )
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    logistic_map = logistic.logistic_map.map_instance(3.8)
    x_0 = list(np.linspace(0.5 - 1e-6, 0.5 + 1e-6, 1000))
    n_steps = 100
    logistic_map.plot(
        x_0,
        n_steps,
        label=f"r={3.8}",
        figsize=(16, 9),
        ax=axs[0],
        edgecolors="aquamarine",
        linewidths=0.1,
        alpha=0.2,
    )
    axs[0].set_title("logistic_map(x, r=3.8)")

    citsigol_map = citsigol.citsigol_map.map_instance(3.8)
    x_0 = [0.5]
    n_steps = 15
    citsigol_map.plot(
        x_0,
        n_steps,
        label="r=3.8",
        figsize=(16, 9),
        fig=fig,
        ax=axs[1],
        linestyles="solid",
        linewidths=0.2,
        edgecolors="aquamarine",
        alpha=0.5,
    )
    axs[1].set_title("citsigol_map(x, r=3.8)")

    plt.show()

    print("Plotting the classic bifurcation diagram of the logistic map...")
    logistic_bifurcation_config = BifurcationDiagramConfig(
        parametrized_map=logistic.logistic_map,
        steps_to_skip=1000,
        initial_values=list(np.linspace(0, 1, 11)),
        n_points=2000,
        resolution=1000,
        parameter_bounds=(-2, 4),
        x_bounds=(-0.5, 1.5),
    )
    logistic_bifurcation_diagram = BifurcationDiagram(
        logistic.logistic_map,
        config=logistic_bifurcation_config,
    )
    logistic_bifurcation_diagram.ax.set_title("Logistic Map Bifurcation Diagram")
    logistic_bifurcation_diagram.display()
    plt.show()

    # plot the citsigol map with a few different r and target values
    print("Plotting the citsigol map's bifurcation diagram...")
    citsigol_bifurcation_config = BifurcationDiagramConfig(
        parametrized_map=citsigol.citsigol_map,
        steps_to_skip=12,
        initial_values=list(np.linspace(0, 1, 5)),
        n_points=2000,
        resolution=1000,
        parameter_bounds=(3, 4),
        x_bounds=(0, 1),
        max_steps=1000,
    )
    citsigol_bifurcation_diagram = BifurcationDiagram(
        citsigol.citsigol_map,
        config=citsigol_bifurcation_config,
    )
    citsigol_bifurcation_diagram.ax.set_title("Citsigol Map Bifurcation Diagram")
    citsigol_bifurcation_diagram.display()
    plt.show()

    print("Plotting a novel, user-defined map...")
    r_stable = 3.4
    r_chaotic = 3.8
    my_map = citsigol.Map(  # define whatever function you want for your map here
        lambda values: [
            logistic.logistic_function([x], r_chaotic)[0]
            if x < 0.5
            else logistic.logistic_function([x], r_stable)[0]
            for x in values
        ]
    )  # The function must take a list of floats and return a list of floats.
    x_0 = list(np.linspace(0.3 - 1e-2, 0.3 + 1e-2, 1000))
    n_steps = 100
    fig, ax = my_map.plot(
        x_0,
        n_steps,
        label=f"r={3.8}",
        figsize=(16, 9),
        edgecolors="aquamarine",
        linewidths=0.1,
        alpha=0.2,
    )
    ax.set_title(
        f"f(x) = (logistic_map(x, {r_chaotic}) if x < 0.5, logistic_map(x, {r_stable}) if x >= 0.5)(x)"
    )
    ax.autoscale()
    plt.show()

    print("Plotting the bifurcation diagram of a novel map...")

    b_default = 5.821
    w1_default = 1.487
    w2_default = 0.2223

    # noinspection PyPep8Naming
    def my_map_function(
        x: list[float],
        A: float,
        B: float = b_default,
        w1: float = w1_default,
        w2: float = w2_default,
    ) -> list[float]:
        """
        See:
        Baghdadi G et al. A chaotic model of sustaining attention problem in attention deficit disorder.
        Commun Nonlinear Sci Numer Simulat (2014), http://dx.doi.org/10.1016/j.cnsns.2014.05.015
        """
        return [B * math.tanh(w1 * val) - A * math.tanh(w2 * val) for val in x]

    # noinspection PyPep8Naming
    my_parametrized_map = citsigol.ParametrizedMap(
        parametrized_function=lambda x, A: my_map_function(
            x, A, b_default, w1_default, w2_default
        ),
        parameter_name="A",
        steps_to_skip=2000,
        initial_values=list(np.linspace(-1, 1, 11)),
        n_points=2000,
        resolution=2000,
        parameter_bounds=(0, 35),
        x_bounds=(-6, 6),
    )
    bifurcation_diagram = my_parametrized_map.bifurcation_diagram(
        marker="_", markersize=0.1
    )
    bifurcation_diagram.ax.set_title(
        f"{b_default}tanh({w1_default}x) - A tanh({w2_default}x)"
    )
    bifurcation_diagram.display()
    plt.show()

    print("Thanks for using citsigol!")


if __name__ == "__main__":
    main()
