"""Demo script for citsigol."""

import matplotlib.pyplot as plt
import mpld3
import numpy as np

import citsigol.logistic


def main() -> None:
    """Console script for citsigol."""
    print("Welcome to citsigol!")
    print("This is a simple demo script.")

    # plot the classic bifurcation diagram of the logistic map
    print("Plotting the classic bifurcation diagram of the logistic map")
    r_values = np.linspace(0, 4.0, 401)
    plt.plot(
        r_values, citsigol.logistic.bifurcation_diagram(r_values), "k.", markersize=0.5
    )
    mpld3.show()


if __name__ == "__main__":
    main()
