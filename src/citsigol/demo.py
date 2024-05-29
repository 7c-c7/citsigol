"""Demo script for citsigol."""

import math

import matplotlib.pyplot as plt
import numpy as np

import citsigol
import citsigol.logistic as logistic


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")

    r_resolution = 2001
    n_steps = 400
    n_skip = 400
    r_limits = (2, 4)
    # plot the classic bifurcation diagram of the logistic map
    print("Plotting the classic bifurcation diagram of the logistic map...")
    r_values = np.linspace(r_limits[0], r_limits[1], r_resolution)
    x_values = logistic.bifurcation_diagram(
        r_values, n_skip=n_skip, n_steps=n_steps, track_progress=True, tol=1e-6
    )
    plt.figure(figsize=(21, 11))
    plt.style.use("dark_background")
    for (
        x,
        r,
    ) in zip(x_values, r_values):
        plt.plot(
            r * np.ones_like(x), x, ".", markersize=0.3, alpha=0.4, color="aquamarine"
        )
    print("\nDone!")

    # plot the stable limit cycles of the logistic map vs the parameter.
    """
    r_values = np.linspace(r_limits[0], r_limits[1], round(r_resolution / 2))
    max_period = 4
    x_values = [logistic.fixed_points(r, max_period) for r in r_values]
    plt.xlabel("r")
    plt.ylabel("x")
    for x, r in zip(x_values, r_values):
        plt.plot(
            r * np.ones_like(x), x, ".", markersize=0.3, alpha=0.6, color="aquamarine"
        )
    """

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
        exes = citsigol.CitsigolMap(r).sequence([max(0.1, r / 5)], n_iterations)
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
