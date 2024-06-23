"""Tests for `citsigol` package."""

import pytest

import citsigol
from citsigol import bifurcation


def test_citsigol_map():
    """Test the citsigol function call."""
    citsigol_map = citsigol.citsigol_map.map_instance(1.0)
    assert citsigol_map([0.125]) == pytest.approx(
        [
            0.5 * (1 - (1 - 0.125 * 4.0 / 1) ** 0.5),
            0.5 * (1 + (1 - 0.125 * 4.0 / 1) ** 0.5),
        ]
    )
    assert citsigol_map([0.125])[1] == pytest.approx(
        0.5 * (1 + (1 - 0.125 * 4.0 / 1) ** 0.5)
    )
    assert citsigol_map([0.125])[0] == pytest.approx(
        0.5 * (1 - (1 - 0.125 * 4.0 / 1) ** 0.5)
    )
    assert citsigol_map([0.125, 0.25]) == pytest.approx(
        [
            0.5 * (1 - (1 - 0.125 * 4.0 / 1) ** 0.5),
            0.5 * (1 + (1 - 0.125 * 4.0 / 1) ** 0.5),
            0.5 * (1 - (1 - 0.25 * 4.0 / 1) ** 0.5),
            0.5 * (1 + (1 - 0.25 * 4.0 / 1) ** 0.5),
        ]
    )
    assert not citsigol_map([0.5])  # empty list since this is out of bounds


def test_citsigol_sequence():
    """Test the citsigol sequence function."""
    citsigol_map = citsigol.citsigol_map.map_instance(3.8)
    sequence = list(citsigol_map.sequence([0.125], 10))
    for i, x_n in enumerate(sequence[1:]):
        assert x_n == citsigol_map(sequence[i])


def test_citsigol_bifurcation_diagram():
    """Test the citsigol bifurcation diagram."""
    citsigol_map = citsigol.citsigol_map
    config = bifurcation.BifurcationDiagramConfig(
        parametrized_map=citsigol_map,
    )
    bifurcation_diagram = citsigol_map.bifurcation_diagram(config=config)
    assert bifurcation_diagram.figure
    assert bifurcation_diagram.ax
    assert bifurcation_diagram.parametrized_map is citsigol_map


def test_novel_map():
    """Test the creation of a novel map."""

    def novel_map_function(x: list[float], a: float, b: float = 1.0) -> list[float]:
        return [a * x_val + b for x_val in x]

    novel_map = citsigol.ParametrizedMap(novel_map_function)
    assert novel_map([1.0], 2.0, 3.0) == [5.0]
    assert novel_map([1.0], 2.0) == [3.0]
    map_2_3 = novel_map.map_instance(2.0, 3.0)
    assert map_2_3([1.0]) == [5.0]
    assert map_2_3([5.0]) == [13.0]
    assert list(map_2_3.sequence([1.0], 2)) == [[5.0], [13.0]]
    map_2 = novel_map.map_instance(2.0)
    assert map_2([1.0]) == [3.0]
    assert list(map_2.sequence([1.0], 2)) == [[3.0], [7.0]]
