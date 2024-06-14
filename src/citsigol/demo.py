"""Demo script for citsigol."""

import matplotlib.pyplot as plt
import numpy as np

import citsigol
import citsigol.logistic as logistic
from citsigol.bifurcation import BifurcationDiagram, BifurcationDiagramConfig


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")

    # plot the classic bifurcation diagram of the logistic map
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

    # plot the citsigol map with a few different r and target values
    print("Plotting the citsigol map's bifurcation diagram...")
    citsigol_bifurcation_config = BifurcationDiagramConfig(
        map_class=citsigol.CitsigolMap,
        steps_to_skip=10,
        initial_values=[0.1],
        n_points=100,
        resolution=100,
        r_bounds=(3, 4),
        x_bounds=(0, 1),
        max_steps=100,
    )
    citsigol_bifurcation_diagram = BifurcationDiagram(
        citsigol.CitsigolMap,
        config=citsigol_bifurcation_config,
    )
    citsigol_bifurcation_diagram.display()
    plt.show()


if __name__ == "__main__":
    main()
