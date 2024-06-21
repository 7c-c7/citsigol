"""Tests for `citsigol` package."""

import pytest

import citsigol


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
