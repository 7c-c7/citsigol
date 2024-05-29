"""Demo script for citsigol."""

import math

import matplotlib.pyplot as plt

import citsigol
import citsigol.logistic as logistic
from citsigol.bifurcation import BifurcationDiagram


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")

    r_resolution = 501
    n_points = 1100
    n_skip = 1000
    r_limits = (3.544, 4)
    # plot the classic bifurcation diagram of the logistic map
    print("Plotting the classic bifurcation diagram of the logistic map...")
    bifurcation_diagram = BifurcationDiagram(
        logistic.LogisticMap,
        steps_to_skip=n_skip,
        initial_values=[0.53],
        n_points=n_points,
        resolution=r_resolution,
        r_bounds=r_limits,
    )
    bifurcation_diagram.display()

    # plot the citsigol map with a few different r and target values
    print("Plotting the citsigol map with a few different r and target values...")
    n_iterations = 10
    r_values = [0.5, 1.5, 2.5, math.pi, 3.5, 3.8]
    plt.figure(figsize=(21, 11))
    plt.title("Citsigol Map with Compass")
    plt.xlabel("n")
    plt.ylabel("x_n")
    for r in r_values:
        print(f"r={r:.5f}")
        exes = list(citsigol.CitsigolMap(r).sequence([max(0.1, r / 5)], n_iterations))
        steps = [step for i, x in enumerate(exes) for step in len(x) * [i]]
        plot_exes = [x for xs in exes for x in xs]
        plt.plot(steps, plot_exes, ".", label=f"r={r}")
    plt.legend()
    plt.title("Citsigol Map with Compass")
    plt.xlabel("n")
    plt.ylabel("x_n")
    plt.show()


if __name__ == "__main__":
    main()
