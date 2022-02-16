import numpy as np
from numpy import testing
import pytest

from .basis import Basis
from .grid import Grid
from .cell import Cell


def _parabolic(n, d2=False):
    grid = [np.linspace(-.5, .5, n, endpoint=False)] * 2
    if not d2:
        grid.append(np.array((0,)))
    mg = np.meshgrid(*grid, indexing='ij')
    data = sum(i ** 2 for i in mg) ** .5

    return Grid(Basis.orthorhombic((1, 1) if d2 else (1, 1, 1)), grid, data)


grid1 = _parabolic(1)
cell1 = grid1.as_cell()
grid50 = _parabolic(50)
cell50 = grid50.as_cell()
grid50_2d = _parabolic(50, True)
cell50_2d = grid50_2d.as_cell()
cell = Cell(Basis.orthorhombic((1, 1)), [[.2, .3], [.6, .1], [0, .5]], [1, 2, 3])


@pytest.mark.parametrize("grid", (grid1, cell1, grid50, cell50, grid50_2d, cell50_2d, cell))
def test_triangulation(grid):
    tri = grid.compute_triangulation()
    testing.assert_allclose(tri.weights.sum(), 1)


@pytest.mark.parametrize("grid", (grid50, cell50))
def test_default(grid):
    d = grid.tetrahedron_density((-.1, 0, .1, .2))
    testing.assert_allclose(d, (0, 0, 2 * np.pi * 0.1, 2 * np.pi * 0.2), rtol=1e-2)


def test_resolved():
    d = grid50.tetrahedron_density((-.1, 0, .1, .2), resolved=True)
    testing.assert_equal(d.values.shape, (*grid50.grid_shape, 4))
    testing.assert_allclose(d.values.sum(axis=(0, 1, 2)), (0, 0, 2 * np.pi * 0.1, 2 * np.pi * 0.2), rtol=1e-2)


@pytest.mark.parametrize("grid", (grid50, cell50))
def test_weighted(grid):
    d = grid.tetrahedron_density((-.1, 0, .1, .2), weights=np.ones_like(grid.values) * .5)
    testing.assert_allclose(d, (0, 0, np.pi * 0.1, np.pi * 0.2), rtol=1e-2)


def test_fail():
    g = grid50.copy(
        vectors=grid50.vectors[:2, :2],
        coordinates=grid50.coordinates[:2],
        values=grid50.values[:, :, 0, ...],
    )
    with pytest.raises(ValueError):
        g.tetrahedron_density((-.1, 0, .1, .2))
