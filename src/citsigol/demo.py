"""Demo script for citsigol."""

import math

import matplotlib.pyplot as plt
import numpy as np

import citsigol.logistic as logistic
import citsigol


def main() -> None:
    """Demo script for citsigol."""
    print("Welcome to citsigol!")

    """
    # plot the classic bifurcation diagram of the logistic map
    print("Plotting the classic bifurcation diagram of the logistic map...")
    r_values = np.linspace(3.2, 4, 801)
    x_values = logistic.bifurcation_diagram(
        r_values, n_skip=100, n_steps=100, track_progress=True, tol=1e-5
    )
    plt.figure(figsize=(21, 11))
    for (
        x,
        r,
    ) in zip(x_values, r_values):
        plt.plot(r * np.ones_like(x), x, "k.", markersize=0.5)
    print("Done!")
    """

    # plot the stable limit cycles of the logistic map vs the parameter.
    r_values = np.linspace(0, 4, 8001)
    max_period = 4
    x_values = [logistic.fixed_points(r, max_period) for r in r_values] + [logistic.fixed_points(r, 3) for r in r_values]
    r_values = np.array(list(np.linspace(0, 4, 8001)) * 2)
    plt.figure(figsize=(21, 11))
    plt.title(f"Stable Limit Cycles of the Logistic Map up to Period {max_period}")
    plt.xlabel("r")
    plt.ylabel("x")
    for x, r in zip(x_values, r_values):
        plt.plot(r * np.ones_like(x), x, "k.", markersize=0.5)

    # plot the citsigol map with a few different r and target values
    print("Plotting the citsigol map with a few different r and target values...")
    n_iterations = 200
    r_values = [0.5, 1.5, 2.5, math.pi, 3.5, 3.8]
    target_values = [0.25, 0.6, 1 / math.pi]
    exes = []
    for target in target_values:
        plt.figure(figsize=(21, 11))
        plt.title("Citsigol Map with Compass")
        plt.xlabel("n")
        plt.ylabel("x_n")
        for r in r_values:
            print(f"r={r:.5f}, target={target:.5f}")
            compass = citsigol.Seeker(target)
            steps = list(range(n_iterations + 1))
            exes = citsigol.Citsigol(r).sequence(max(0.1, r / 5), n_iterations, compass)
            plt.plot(steps[: len(exes)], exes, label=f"r={r}")
        plt.legend()
        plt.title("Citsigol Map with Compass")
        plt.plot([0, len(exes)], [target, target], "k--", label="target")
        plt.xlabel("n")
        plt.ylabel("x_n")
    plt.show()


if __name__ == "__main__":
    main()
