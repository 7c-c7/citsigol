"""Demo script for citsigol."""

import matplotlib.pyplot as plt
import numpy as np

import citsigol
import citsigol.logistic as logistic
from citsigol.bifurcation import BifurcationDiagram, BifurcationDiagramConfig


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")

    logistic_map_stable = logistic.LogisticMap(3.4)
    logistic_map_chaotic = logistic.LogisticMap(3.8)

    print(
        "Plotting examples of the logistic map and the citsigol map over several iterations..."
    )
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    logistic_map = logistic.LogisticMap(3.8)
    x_0 = list(np.linspace(0.5 - 1e-6, 0.5 + 1e-6, 1000))
    n_steps = 100
    logistic_map.plot(
        x_0,
        n_steps,
        label=f"r={logistic_map.r}",
        figsize=(16, 9),
        ax=axs[0],
        edgecolors="aquamarine",
        linewidths=0.1,
        alpha=0.2,
    )
    axs[0].set_title(f"{logistic_map}")

    citsigol_map = citsigol.CitsigolMap(3.8)
    x_0 = [0.5]
    n_steps = 15
    citsigol_map.plot(
        x_0,
        n_steps,
        label=f"r={citsigol_map.r}",
        figsize=(16, 9),
        fig=fig,
        ax=axs[1],
        linestyles="solid",
        linewidths=0.2,
        edgecolors="aquamarine",
        alpha=0.5,
    )
    axs[1].set_title(f"{citsigol_map}")

    plt.show()

    print("Plotting the classic bifurcation diagram of the logistic map...")
    logistic_bifurcation_config = BifurcationDiagramConfig(
        map_class=logistic.LogisticMap,
        steps_to_skip=300,
        initial_values=list(np.linspace(0, 1, 11)),
        n_points=2000,
        resolution=1000,
        r_bounds=(-2, 4),
        x_bounds=(-0.5, 1.5),
    )
    logistic_bifurcation_diagram = BifurcationDiagram(
        logistic.LogisticMap,
        config=logistic_bifurcation_config,
    )
    logistic_bifurcation_diagram.display()
    logistic_bifurcation_diagram.ax.set_title("Logistic Map Bifurcation Diagram")
    plt.show()

    # plot the citsigol map with a few different r and target values
    print("Plotting the citsigol map's bifurcation diagram...")
    citsigol_bifurcation_config = BifurcationDiagramConfig(
        map_class=citsigol.CitsigolMap,
        steps_to_skip=12,
        initial_values=list(np.linspace(0, 1, 5)),
        n_points=2000,
        resolution=1000,
        r_bounds=(3, 4),
        x_bounds=(0, 1),
        max_steps=1000,
    )
    citsigol_bifurcation_diagram = BifurcationDiagram(
        citsigol.CitsigolMap,
        config=citsigol_bifurcation_config,
    )
    citsigol_bifurcation_diagram.display()
    logistic_bifurcation_diagram.ax.set_title("Citsigol Map Bifurcation Diagram")
    plt.show()

    print("Plotting a novel, user-defined map...")
    my_map = citsigol.Map(  # define whatever function you want for your map here
        lambda values: [
            logistic_map_chaotic([x])[0] if x < 0.5 else logistic_map_stable([x])[0]
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
        f"f(x) = ({logistic_map_chaotic} if x < 0.5, {logistic_map_stable} if x >= 0.5)(x)"
    )
    ax.autoscale()

    plt.show()

    print("Thanks for using citsigol!")


if __name__ == "__main__":
    main()
