import numpy as np
from numpy import testing
import pytest
from unittest import TestCase
import pickle

from .basis import Basis
from .grid import Grid


def test_init_0():
    x = np.linspace(0, 1, 10, endpoint=False)
    y = np.linspace(0, 1, 20, endpoint=False)
    z = np.linspace(0, 1, 30, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    data = xx * 2 + yy * 2 + zz * 2
    c = Grid(
        Basis.orthorhombic((1, 1, 1)),
        (x, y, z),
        data,
    )
    assert len(c.coordinates) == 3
    testing.assert_equal(c.coordinates[0], x)
    testing.assert_equal(c.coordinates[1], y)
    testing.assert_equal(c.coordinates[2], z)
    testing.assert_equal(c.values.shape, (10, 20, 30))


def test_init_fail_0():
    x = np.linspace(0, 1, 10, endpoint=False)
    y = np.linspace(0, 1, 10, endpoint=False)
    z = np.linspace(0, 1, 10, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    data = xx ** 2 + yy ** 2 + zz ** 2
    basis = Basis.orthorhombic((1, 1, 1))

    with pytest.raises(ValueError):
        Grid(basis, (x, y), data)

    with pytest.raises(ValueError):
        Grid(basis, (x, y, z, z), data)

    with pytest.raises(ValueError):
        Grid(basis, ((x, x), (y, y), (z, z)), data)

    with pytest.raises(ValueError):
        Grid(basis, (x, y, z), data[0])


def test_init_1():
    x = np.linspace(0, 1, 10, endpoint=False)
    y = np.linspace(0, 1, 20, endpoint=False)
    z = np.linspace(0, 1, 30, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    data = xx ** 2 + yy ** 2 + zz ** 2

    with pytest.raises(ValueError):
        Grid(
            Basis.orthorhombic((1, 1, 1)),
            (x, x, x),
            data,
        )


class GridTest(TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 1, 2, endpoint=False)
        y = np.linspace(0, 1, 3, endpoint=False)
        z = np.linspace(0, 1, 4, endpoint=False)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx ** 2 + yy ** 2 + zz ** 2
        cls.empty = Basis.orthorhombic((1, 2, 3))
        cls.grid = Grid(cls.empty, (x, y, z), data)

    def test_pickle(self):
        c = pickle.loads(pickle.dumps(self.grid))
        testing.assert_equal(c.vectors, self.grid.vectors)
        testing.assert_equal(c.coordinates, self.grid.coordinates)
        testing.assert_equal(c.values, self.grid.values)

    def test_eq(self):
        assert self.grid == self.grid.copy()
        assert self.grid == self.grid.copy(coordinates=self.grid.coordinates)
        assert self.grid != self.grid.copy(coordinates=tuple(i + 3.14 for i in self.grid.coordinates))
        assert self.grid != self.grid.copy(values=self.grid.values + 3.14)

    def test_round(self):
        g = self.grid.rounded(1)
        testing.assert_equal(g.vectors, [(1, 0, 0), (0, 2, 0), (0, 0, 3)])
        testing.assert_equal(g.coordinates, [(0, .5), (0, .3, .7), (0, .2, .5, .8)])
        testing.assert_equal(g.values, self.grid.values)

    def test_size(self):
        assert self.grid.size == 24

    def test_explicit_coordinates(self):
        c = self.grid.explicit_coordinates()
        assert len(c.shape) == 4
        assert (c[1, :, :, 0] == 0.5).all()
        assert (c[:, 1, :, 1] == 1. / 3).all()
        assert (c[:, :, 1, 2] == 1. / 4).all()

    def test_cartesian(self):
        c = self.grid.cartesian()
        testing.assert_allclose(c[0, 0, 0], (0, 0, 0))
        testing.assert_allclose(c[1, 2, 3], (1. / 2 * 1, 2. / 3 * 2, 3. / 4 * 3))

    def test_select_0(self):
        testing.assert_equal(self.grid.select((0.3, 0.3, 0.3, 0.7, 0.7, 0.7)), (
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))

    def test_select_1(self):
        testing.assert_equal(self.grid.select((0.3, 0, 0, 0.7, 1, 1)), (
            (False, True),
            (True, True, True),
            (True, True, True, True)
        ))

    def test_select_fail(self):
        with self.assertRaises(ValueError):
            self.grid.select(0.3, 0.3, 0.7, 0.7)

    def test_apply(self):
        c = self.grid.apply((
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))
        testing.assert_allclose(c.coordinates[0], (0.5,))
        testing.assert_allclose(c.coordinates[1], (1. / 3, 2. / 3))
        testing.assert_allclose(c.coordinates[2], (0.5,))
        testing.assert_allclose(c.values, (
            ((0.5 + 1. / 9,), (0.5 + 4. / 9,)),
        ))

    def test_discard(self):
        c = self.grid.discard((
            (False, True),
            (False, True, True),
            (False, False, True, False)
        ))
        testing.assert_allclose(c.coordinates[0], (0.,))
        testing.assert_allclose(c.coordinates[1], (0.,))
        testing.assert_allclose(c.coordinates[2], (0., .25, .75))
        testing.assert_allclose(c.values, (
            ((0., .25 ** 2, .75 ** 2),),
        ))

    def test_cut_0(self):
        c = self.grid.cut(0.3, 0.3, 0.3, 0.7, 0.7, 0.7)
        testing.assert_allclose(c.coordinates[0], ((0.5 - 0.3) / 0.4,))
        testing.assert_allclose(c.coordinates[1], ((1. / 3 - 0.3) / 0.4, (2. / 3 - 0.3) / 0.4))
        testing.assert_allclose(c.coordinates[2], ((0.5 - 0.3) / 0.4,))
        testing.assert_allclose(c.values, (
            ((0.5 + 1. / 9,), (0.5 + 4. / 9,)),
        ))

    def test_cut_1(self):
        c = self.grid.cut(0.3, 0, 0, 0.7, 1, 1)
        testing.assert_allclose(c.coordinates[0], ((0.5 - 0.3) / 0.4,))
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[2])
        testing.assert_allclose(c.values, self.grid.values[[1], ...])

    def test_merge(self):
        x = np.linspace(0, 1, 2, endpoint=False)
        y = np.linspace(0, 1, 3, endpoint=False)
        z = np.linspace(0, 1, 4, endpoint=False) + 0.1
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx ** 2 + yy ** 2 + zz ** 2
        grid = Grid(self.grid, (x, y, z), data)
        grid_merged = grid.merge(self.grid)

        z = np.sort(np.concatenate((z, self.grid.coordinates[2])))
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx ** 2 + yy ** 2 + zz ** 2
        grid_ref = Grid(self.grid, (x, y, z), data)

        testing.assert_allclose(grid_merged.vectors, grid_ref.vectors)
        for i in range(3):
            testing.assert_allclose(grid_merged.coordinates[i], grid_ref.coordinates[i])
        testing.assert_allclose(grid_merged.values, grid_ref.values)

    def test_normalized(self):
        coords = tuple(c + i / 10 + 0.1 for i, c in enumerate(self.grid.coordinates))
        c = self.grid.copy(coordinates=coords).normalized(sort=True)

        testing.assert_allclose(c.coordinates[0], (.1, .6))
        testing.assert_allclose(c.coordinates[1], (.2, 1. / 3 + .2, 2. / 3 + .2))
        testing.assert_allclose(c.coordinates[2], (.75 - .7, .3, .25 + .3, .5 + .3))
        testing.assert_allclose(c.values, self.grid.values[:, :, (3, 0, 1, 2)])

    def test_isolated_0(self):
        c = self.grid.isolated((1, 2, 3), units='crystal')

        testing.assert_allclose(c.coordinates[0], (self.grid.coordinates[0] + 0.5) / 2)
        testing.assert_allclose(c.coordinates[1], (self.grid.coordinates[1] + 1) / 3)
        testing.assert_allclose(c.coordinates[2], (self.grid.coordinates[2] + 1.5) / 4)
        testing.assert_allclose(c.values, self.grid.values)

    def test_isolated_1(self):
        c = self.grid.isolated((1, 4, 9), units='cartesian')

        testing.assert_allclose(c.coordinates[0], (self.grid.coordinates[0] + 0.5) / 2)
        testing.assert_allclose(c.coordinates[1], (self.grid.coordinates[1] + 1) / 3)
        testing.assert_allclose(c.coordinates[2], (self.grid.coordinates[2] + 1.5) / 4)
        testing.assert_allclose(c.values, self.grid.values)

    def test_isolated_fail(self):
        with self.assertRaises(ValueError):
            self.grid.isolated((1, 2, 3), units='unknown')

    def test_stack_0(self):
        vectors = self.grid.vectors.copy()
        vectors[2, 2] = 6
        another = self.grid.copy(vectors=vectors)
        c = Grid.stack(self.grid, another, vector=2)

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], np.concatenate((
            self.grid.coordinates[2] / 3, another.coordinates[2] * 2 / 3 + 1. / 3
        )))
        testing.assert_allclose(c.values, np.concatenate((
            self.grid.values, another.values
        ), axis=2))

    def test_stack_1(self):
        x = np.linspace(0, 1, 2, endpoint=False)
        y = np.linspace(0, 1, 3, endpoint=False)
        z = np.linspace(0.5, 0.7, 15)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx ** 2 + yy ** 2 + zz ** 2
        another = Grid(Basis.orthorhombic((1, 2, 6)), (x, y, z), data)
        c = Grid.stack(self.grid, another, vector=2)

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], np.concatenate((
            self.grid.coordinates[2] / 3, another.coordinates[2] * 2 / 3 + 1. / 3
        )))
        testing.assert_allclose(c.values, np.concatenate((
            self.grid.values, another.values
        ), axis=2))

    def test_stack_2(self):
        c = Grid.stack(self.grid, self.empty, vector=2)

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[0])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[2] / 2)
        testing.assert_allclose(c.values, self.grid.values)

    def test_stack_fail_0(self):
        x = np.linspace(0, 1, 3)[:-1]
        y = np.linspace(0, 1, 3)[:-1]
        z = np.linspace(0, 1, 5)[:-1]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx ** 2 + yy ** 2 + zz ** 2
        another = Grid(self.grid, (x, y, z), data)
        with self.assertRaises(ValueError):
            self.grid.stack(another, vector=0)

    def test_repeated_0(self):
        rp = self.grid.repeated(1, 1, 1)
        testing.assert_allclose(rp.vectors, self.grid.vectors)
        assert len(rp.coordinates) == len(self.grid.coordinates)
        for i in range(len(rp.coordinates)):
            testing.assert_allclose(rp.coordinates[i], self.grid.coordinates[i])
        testing.assert_equal(rp.values, self.grid.values)

    def test_repeated_1(self):
        rp = self.grid.repeated(2, 2, 1)
        nv = self.grid.vectors.copy()
        nv[0:2, :] *= 2
        testing.assert_allclose(rp.vectors, nv)
        assert len(rp.coordinates) == len(self.grid.coordinates)

        testing.assert_allclose(rp.coordinates[0], np.linspace(0, 1, 4, endpoint=False))
        testing.assert_allclose(rp.coordinates[1], np.linspace(0, 1, 6, endpoint=False))
        testing.assert_allclose(rp.coordinates[2], np.linspace(0, 1, 4, endpoint=False))
        v = np.concatenate((self.grid.values,) * 2, axis=0)
        v = np.concatenate((v,) * 2, axis=1)
        testing.assert_equal(rp.values, v)

    def test_rv(self):
        c = self.grid.transpose_vectors(2, 1, 0)

        testing.assert_allclose(c.coordinates[0], self.grid.coordinates[2])
        testing.assert_allclose(c.coordinates[1], self.grid.coordinates[1])
        testing.assert_allclose(c.coordinates[2], self.grid.coordinates[0])
        testing.assert_allclose(c.values, self.grid.values.swapaxes(0, 2))

    def test_as_unitCell(self):
        reg = self.grid.as_cell()

        testing.assert_equal(reg.coordinates.shape, (24, 3))
        testing.assert_equal(reg.values.shape, (24,))

        for i, x in enumerate(self.grid.coordinates[0]):
            for j, y in enumerate(self.grid.coordinates[1]):
                for k, z in enumerate(self.grid.coordinates[2]):

                    found = False

                    for l in range(reg.coordinates.shape[0]):
                        if tuple(reg.coordinates[l]) == (x, y, z):
                            found = True
                            assert self.grid.values[i, j, k] == reg.values[l]
                            break

                    if not found:
                        raise AssertionError("Coordinate {} {} {} not found".format(x, y, z))

    def test_back_forth(self):
        c = self.grid.as_cell().as_grid()

        testing.assert_equal(c.coordinates, self.grid.coordinates)
        testing.assert_equal(c.values, self.grid.values)

    def test_interpolate_to_cell_0(self):
        i = self.grid.interpolate_to_cell(((0, 0, 0), (1. / 2, 1. / 3, 1. / 4)), periodic=False)

        testing.assert_equal(i.coordinates, ((0, 0, 0), (1. / 2, 1. / 3, 1. / 4)))
        testing.assert_equal(i.values, (0, 1. / 4 + 1. / 9 + 1. / 16))

    def test_interpolate_to_cell_1(self):
        i = self.grid.interpolate_to_cell(((0, 0, 0), (1. / 2, 1. / 3, 1. / 4), (-1. / 2, -2. / 3, -3. / 4)))

        testing.assert_equal(i.coordinates, ((0, 0, 0), (1. / 2, 1. / 3, 1. / 4), (-1. / 2, -2. / 3, -3. / 4)))
        testing.assert_equal(i.values, (0, 1. / 4 + 1. / 9 + 1. / 16, 1. / 4 + 1. / 9 + 1. / 16))

    def test_interpolate_to_cell_2(self):
        c = self.grid.copy(values=self.grid.values[..., None] * np.array((1, 2))[None, None, None, :])
        i = c.interpolate_to_cell(((0, 0, 0), (1. / 2, 1. / 3, 1. / 4)), periodic=False)

        testing.assert_equal(i.coordinates, ((0, 0, 0), (1. / 2, 1. / 3, 1. / 4)))
        testing.assert_equal(i.values, np.array((0, 1. / 4 + 1. / 9 + 1. / 16))[:, None] * ((1, 2),))

    def test_interpolate_to_cell_3(self):
        c = self.grid.copy(values=self.grid.values[..., None] * np.array((1, 2))[None, None, None, :])
        i = c.interpolate_to_cell(((0, 0, 0), (1. / 2, 1. / 3, 1. / 4), (-1. / 2, -2. / 3, -3. / 4)))

        testing.assert_equal(i.coordinates, ((0, 0, 0), (1. / 2, 1. / 3, 1. / 4), (-1. / 2, -2. / 3, -3. / 4)))
        testing.assert_equal(i.values, np.array((0, 1. / 4 + 1. / 9 + 1. / 16, 1. / 4 + 1. / 9 + 1. / 16))[:,
                                       None] * ((1, 2),))

    def test_interpolate_0(self):
        i = self.grid.interpolate_to_grid(self.grid.coordinates, periodic=False)

        testing.assert_equal(i.coordinates, self.grid.coordinates)
        testing.assert_equal(i.values, self.grid.values)

    def test_interpolate_error_0(self):
        with self.assertRaises(ValueError):
            self.grid.interpolate_to_grid(((0, 0, 0), (1. / 2, 1. / 3, 1. / 4), (-1. / 2, -2. / 3, -3. / 4)), periodic=False)

    def test_interpolate_periodic(self):
        x = np.linspace(0, 1, 2, endpoint=False)
        y = np.linspace(0, 1, 2, endpoint=False)
        z = np.linspace(0, 1, 2, endpoint=False)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data = xx
        grid = Grid(Basis.orthorhombic((1, 2, 3)), (x, y, z), data)
        interpolated = grid.interpolate_to_array(((.25, 0, 0), (.75, 0, 0)), periodic=True)
        testing.assert_equal(interpolated, (.25, .25))

    def test_interpolate_path_2D(self):
        x = np.linspace(0, 1, 10, endpoint=False)
        y = np.linspace(0, 1, 10, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        data = (xx - .5) ** 2 + (yy - .5) ** 2
        grid = Grid(Basis.orthorhombic((1, 1)), (x, y), data)
        path = np.array((
            (0, 0),
            (0, 1),
            (1, 0),
        ))
        cell = grid.interpolate_to_path(path, 100)
        # Check values
        testing.assert_allclose(cell.values, ((cell.coordinates - .5) ** 2).sum(axis=-1), atol=1e-2)

        # Check if all coordinates are on the path
        A = path[None, :-1, :]
        B = path[None, 1:, :]
        C = cell.coordinates[:, None, :]
        Ax, Ay, Bx, By, Cx, Cy = A[..., 0], A[..., 1], B[..., 0], B[..., 1], C[..., 0], C[..., 1]
        area = Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)
        testing.assert_equal(np.any(area < 1e-14, axis=1), True)

        # Check if sum of spacings is equal to total path length
        assert abs(cell.distances(np.arange(cell.size)).sum() - 1 - 2. ** .5) < 1e-14

        # Check if spacings are uniform
        testing.assert_allclose(cell.distances(np.arange(cell.size)), (1 + 2. ** .5) / (cell.size - 1),
                                rtol=2. / cell.size)

    def test_serialization(self):
        serialized = self.grid.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.grid.vectors.tolist(),
            meta={},
            type="pycoordinates.grid.Grid",
            coordinates=tuple(i.tolist() for i in self.grid.coordinates),
            values=self.grid.values.tolist(),
        ))
        assert self.grid == Grid.from_state_dict(serialized)
