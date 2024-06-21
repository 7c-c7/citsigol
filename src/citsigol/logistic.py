"""
Logistic map and basic related functions.
"""

from citsigol import ParametrizedMap


def logistic_function(x: list[float], r: float) -> list[float]:
    """
    Return the logistic function applied to a list of values.

    Parameters
    ----------
    x : list[float]
        List of values to apply the logistic function to.
    r : float
        Parameter of the logistic map.

    Returns
    -------
    list[float]
        List of values after applying the logistic function.
    """
    return [r * value * (1 - value) for value in x]


logistic_map = ParametrizedMap(
    parametrized_function=logistic_function,
    parameter_name="r",
    steps_to_skip=100,
    n_points=100,
    x_bounds=(0, 1),
    parameter_bounds=(0, 4),
    resolution=1000,
    max_steps=100_000,
)
