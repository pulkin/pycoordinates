import numpy as np
from numpy import testing
from unittest import TestCase
import pytest

from .basis import Basis
from .grid import Grid

N = 50
x = np.linspace(-.5, .5, N, endpoint=False)
y = np.linspace(-.5, .5, N, endpoint=False)
z = np.array((0,))
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
data = (xx ** 2 + yy ** 2 + zz ** 2) ** .5

grid = Grid(Basis.orthorhombic((1, 1, 1)), (x, y, z), data)
cell = grid.as_cell()


@pytest.mark.parametrize("grid", (grid, cell))
def test_td_0(grid):
    d = grid.tetrahedron_density((-.1, 0, .1, .2))
    testing.assert_allclose(d, (0, 0, 2 * np.pi * 0.1, 2 * np.pi * 0.2), rtol=1e-2)


def test_td_1():
    d = grid.tetrahedron_density((-.1, 0, .1, .2), resolved=True)
    testing.assert_equal(d.values.shape, (50, 50, 1, 1, 4))
    testing.assert_allclose(d.values.sum(axis=0).sum(axis=0).sum(axis=0)[0],
                            (0, 0, 2 * np.pi * 0.1, 2 * np.pi * 0.2), rtol=1e-2)


@pytest.mark.parametrize("grid", (grid, cell))
def test_td_2(grid):
    d = grid.tetrahedron_density((-.1, 0, .1, .2), weights=np.ones_like(grid.values) * .5)
    testing.assert_allclose(d, (0, 0, np.pi * 0.1, np.pi * 0.2), rtol=1e-2)


def test_td_fail_0():
    g = grid.copy(
        vectors=grid.vectors[:2, :2],
        coordinates=grid.coordinates[:2],
        values=grid.values[:, :, 0, ...],
    )
    with pytest.raises(ValueError):
        g.tetrahedron_density((-.1, 0, .1, .2))
