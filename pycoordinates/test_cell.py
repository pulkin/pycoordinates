import pytest

from .basis import Basis
from .cell import Cell

from numpy import testing
import numpy as np
from unittest import TestCase
import pickle


def test_init_0():
    n = Cell(
        Basis.orthorhombic((1e-10, 1e-10, 1e-10)),
        ((.25, .5, .5), (.5, .25, .5), (.5, .5, .25), (.25, .25, .25)),
        ((1,), (2,)) * 2,
    )
    testing.assert_equal(n.coordinates, ((.25, .5, .5), (.5, .25, .5), (.5, .5, .25), (.25, .25, .25)))
    testing.assert_equal(n.values, ((1,), (2,)) * 2)


def test_init_1():
    n = Cell.from_cartesian(
        Basis.orthorhombic((1, 2, 3)),
        [(.5, .5, .5)],
        ['N'],
    )
    testing.assert_equal(n.coordinates, ((.5, .25, 1. / 6),))


def test_init_fail_size_0():
    with pytest.raises(ValueError):
        Cell(
            Basis.orthorhombic((1, 1, 1)),
            [(.25, .5, .5, .5)],
            ['C'],
        )


def test_init_fail_size_1():
    with pytest.raises(ValueError):
        Cell(
            Basis.orthorhombic((1, 1, 1)),
            (
                (.25, .5, .5, .5),
                (.25, .5, .5, .5),
            ),
            ['C'] * 2,
        )


def test_init_fail_size_2():
    with pytest.raises(ValueError):
        Cell(
            Basis.orthorhombic((1, 1, 1)),
            (
                (.25, .5, .5),
                (.25, .5, .5),
            ),
            ['C'] * 3,
        )


def test_init_fail_shape_0():
    with pytest.raises(ValueError):
        Cell(
            Basis.orthorhombic((1, 1, 1)),
            [[(.25, .5, .5)]],
            ['C'],
        )


def test_random_0():
    random = Cell.random(1, {"a": 3, "b": 5})
    testing.assert_allclose(random.vectors, 2 * np.eye(3))
    testing.assert_equal(random.values, ["a"] * 3 + ["b"] * 5)
    assert random.coordinates.shape == (8, 3)
    testing.assert_array_less(random.coordinates, 1)
    testing.assert_array_less(0, random.coordinates)


def test_random_fail_0():
    with pytest.raises(ValueError):
        Cell.random(1, {"a": 3, "b": 5}, shape="unknown")


class CellInitializationTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = a = 2.510e-10
        cls.h = a * (2. / 3.) ** 0.5
        cls.hcpCo_b = Basis.triclinic((a,) * 3, (0.5,) * 3)

    def test_init_0(self):
        hcpCo = Cell(self.hcpCo_b, (0, 0, 0), ['Co'])
        testing.assert_equal(hcpCo.coordinates, [(0, 0, 0)])
        testing.assert_equal(hcpCo.values, ['Co'])
        testing.assert_allclose((hcpCo.vectors ** 2).sum(axis=1), self.a ** 2)

    def test_init_1(self):
        hcpCo = Cell(
            Basis.triclinic((self.a, self.a, self.h * 2), (0., 0., 0.5)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            ['Co'] * 2,
        )
        testing.assert_equal(hcpCo.coordinates, ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)))
        testing.assert_equal(hcpCo.values, ['Co'] * 2)


class TestCell(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c = Cell(
            vectors=np.array([[1, 0, 0], [2, 3, 0], [0, 0, 4]]),
            coordinates=[(.5, .5, 0), (.5, .5, .5)],
            values=["a", "b"],
            meta=dict(
                scalar=3.4,
                str="abc",
                array=np.array((9, 8, 7)),
                list=["1", 1],
            ),
            vectors_inv=np.array([
                (1, 0, 0),
                (-.7, .3, 0),
                (0, 0, .25),
            ])
        )

    def __assert_cells_same__(self, c, r):
        testing.assert_allclose(c.vectors, r.vectors)
        testing.assert_equal(c.coordinates, r.coordinates)
        testing.assert_equal(c.values, r.values)
        cm = dict(c.meta)
        rm = dict(r.meta)
        for k in "scalar", "str", "list", "array":
            testing.assert_equal(cm.pop(k), rm.pop(k), err_msg=k)
        assert len(cm) == 0
        assert len(rm) == 0

    def test_state(self):
        r = Cell.from_state_dict(self.c.state_dict())
        self.__assert_cells_same__(self.c, r)

    def test_prop(self):
        c = self.c
        assert c.size == 2
        testing.assert_equal(c.values_encoded, [0, 1])
        testing.assert_equal(c.values_uq, ["a", "b"])
        testing.assert_equal(c.values_lookup, {"a": 0, "b": 1})
        testing.assert_allclose(c.vectors_inv, [
            (1, 0, 0),
            (-.7, .3, 0),
            (0, 0, .25),
        ])

    def test_delta_0(self):
        testing.assert_allclose(self.c.cartesian_delta(self.c), np.zeros(self.c.size), atol=1e-14)

    def test_delta_1(self):
        other = self.c.copy(coordinates=[(.5, .5, .9), (.5, .5, .5)])
        testing.assert_allclose(self.c.cartesian_delta(other), [.4, 0], atol=1e-14)
        testing.assert_allclose(other.cartesian_delta(self.c), [.4, 0], atol=1e-14)
        testing.assert_allclose(self.c.cartesian_delta(other, pbc=False), [3.6, 0], atol=1e-14)

    def test_cartesian_copy(self):
        other = self.c.cartesian_copy(vectors=self.c.vectors * 2)
        testing.assert_allclose(other.coordinates, self.c.coordinates / 2)


class CellTest(TestCase):

    @staticmethod
    def __co__(a, h, **kwargs):
        return Cell(
            ((a, 0, 0), (.5 * a, .5 * a * 3. ** .5, 0), (0, 0, h)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            ['Co'] * 2,
            **kwargs
        )

    @staticmethod
    def __bs__(a, h, **kwargs):
        return Cell(
            ((a, 0, 0), (.5 * a, .5 * a * 3.**.5, 0), (0, 0, h)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            [3] * 2,
            **kwargs
        )

    @classmethod
    def setUpClass(cls):
        cls.a = 2.51
        cls.h = 2 * cls.a * (2. / 3.) ** 0.5
        cls.cell = CellTest.__co__(cls.a, cls.h)
        cls.cell2 = Cell(
            Basis.triclinic((cls.a, cls.a, cls.a), (0.5, 0.5, 0.5)),
            (0, 0, 0),
            'Co'
        )
        cls.empty = Basis.triclinic((cls.a, cls.a, cls.a), (0.5, 0.5, 0.5))

    def test_eq(self):
        c = self.cell.copy()
        assert self.cell == c
        c = c.copy(coordinates=((3.14, 0., 0.), (1. / 3., 1. / 3., 0.5)))
        assert self.cell != c
        c = c.copy(coordinates=((0, 0., 0.), (1. / 3., 1. / 3., 0.5)))
        assert self.cell == c
        c = c.copy(values=['x', 'Co'])
        assert self.cell != c

    def test_round(self):
        c = self.cell.rounded(1)
        testing.assert_allclose(c.vectors, [(2.5, 0, 0), (1.3, 2.2, 0), (0, 0, 4.1)])
        testing.assert_allclose(c.coordinates, [(0, 0, 0), (0.3, 0.3, 0.5)])

    def test_volume_0(self):
        assert abs(self.cell.volume / (.5 * 3. ** .5 * self.a ** 2 * self.h) - 1) < 1e-7

    def test_volume_1(self):
        assert abs(self.cell.volume / (2 * self.cell2.volume) - 1) < 1e-7

    def test_size(self):
        assert self.cell.size == 2 * self.cell2.size == 2

    def test_copy_0(self):
        cp = self.cell.copy()
        assert cp.vectors is not self.cell.vectors
        assert cp.coordinates is not self.cell.coordinates
        assert cp.values is not self.cell.values
        testing.assert_equal(cp.vectors, self.cell.vectors)
        testing.assert_equal(cp.coordinates, self.cell.coordinates)
        testing.assert_equal(cp.values, self.cell.values)

    def test_cut_0(self):
        cp = self.cell.cut(0., 0., 0., 1., 1., 1.)
        testing.assert_allclose(cp.vectors, self.cell.vectors)
        testing.assert_allclose(cp.coordinates, self.cell.coordinates)
        testing.assert_equal(cp.values, self.cell.values)

    def test_cut_1(self):
        cp = self.cell.cut(0., 0., 0., .5, .5, .5)
        testing.assert_allclose((cp.vectors ** 2).sum(axis=1) * 4, (self.a ** 2, self.a ** 2, self.h ** 2))
        testing.assert_equal(cp.coordinates, ((0., 0., 0.),))
        testing.assert_equal(cp.values, ("Co",))

    def test_cut_2(self):
        cp = self.cell.cut(.25, .25, .25, .75, .75, .75)
        testing.assert_allclose((cp.vectors ** 2).sum(axis=1) * 4, (self.a ** 2, self.a ** 2, self.h ** 2))
        testing.assert_allclose(cp.coordinates, ((1. / 6, 1. / 6, 1. / 2),))
        testing.assert_equal(cp.values, ("Co",))

    def test_cut_3(self):
        cp = self.cell.cut(.25, .75, .75, .75, .25, .25)
        testing.assert_allclose((cp.vectors ** 2).sum(axis=1) * 4, (self.a ** 2, self.a ** 2, self.h ** 2))
        testing.assert_allclose(cp.coordinates, ((1. / 6, 1. / 6, 1. / 2),))
        testing.assert_equal(cp.values, ("Co",))

    def test_cartesian(self):
        testing.assert_allclose(self.cell.cartesian, ((0., 0., 0.), (self.a / 2, self.a / 2 / 3. ** .5, self.h / 2)))

    def test_angles(self):
        supercell = self.cell.repeated(2, 2, 1)
        testing.assert_allclose(supercell.angles(0, 2, 4, 6), (.5,) * 2)
        testing.assert_allclose(supercell.angles(1, 3, 5, 7), (.5,) * 2)
        testing.assert_allclose(supercell.angles((0, 4, 6), (0, 2, 6)), (-.5,) * 2)
        testing.assert_allclose(supercell.angles(np.array(((0, 4, 6), (0, 2, 6)))), (-.5,) * 2)
        with self.assertRaises(ValueError):
            self.cell.angles(((0, 1, 2), (1, 2, 3)), ((2, 3, 4), (2, 4, 0)))
        with self.assertRaises(ValueError):
            self.cell.angles((0, 1, 2, 3), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            self.cell.angles(0, 1)
        with self.assertRaises(ValueError):
            self.cell.angles(1, 1, 1, 1)

    def test_distances_0(self):
        d = (self.a ** 2 / 3 + self.h ** 2 / 4) ** .5
        testing.assert_allclose(self.cell.distances(0, 1, 0, 1), (d,) * 3)
        with self.assertRaises(ValueError):
            self.cell.distances(0)
        testing.assert_allclose(self.cell.distances(), (
            (0, d),
            (d, 0)
        ))

    def test_distances_1(self):
        supercell = self.cell.repeated(2, 2, 1)
        testing.assert_allclose(supercell.distances(0, 2, 4, 6), (self.a,) * 3)
        testing.assert_allclose(supercell.distances(1, 3, 5, 7), (self.a,) * 3)
        testing.assert_allclose(supercell.distances((0, 4), (2, 6)), (self.a,) * 2)
        testing.assert_allclose(supercell.distances(np.array(((0, 6),))), (self.a * 3 ** .5,))
        with self.assertRaises(ValueError):
            self.cell.distances((0,))
        with self.assertRaises(ValueError):
            self.cell.distances((0, 1, 2), (1, 2, 3))
        with self.assertRaises(ValueError):
            self.cell.distances((0, 1, 2, 3), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            self.cell.distances(((0, 1, 2), (1, 2, 3)), ((2, 3, 4), (3, 4, 0)))

    def test_distances_2(self):
        d = np.linalg.norm(self.cell.vectors[:2].sum(axis=0))
        h = np.linalg.norm(self.cell.vectors[2])

        def _s(a, b):
            return (a ** 2 + b ** 2) ** .5

        testing.assert_allclose(
            self.cell.distances(other=self.cell.copy(coordinates=self.cell.coordinates + 0.01)),
            [[0.01 * _s(d, h), _s(d * (1./3 + 0.01), h * (.5 + 0.01))], [_s(d * (1./3 - 0.01), h * (.5 - 0.01)), 0.01 * _s(d, h)]],
        )

    def test_distances_sparse_0(self):
        d = np.linalg.norm(self.cell.vectors.sum(axis=0))
        distance_matrix = self.cell.distances(cutoff=0.1, other=self.cell.copy(coordinates=self.cell.coordinates + 0.01))
        assert distance_matrix.nnz == 2
        testing.assert_allclose(distance_matrix.todense(), d * 0.01 * np.eye(2))

    def test_centered(self):
        centered = self.cell.centered()
        testing.assert_equal(centered.vectors, self.cell.vectors)
        testing.assert_allclose(centered.coordinates, [
            (1./3, 1./3, .25),
            (2./3, 2./3, .75),
        ])
        testing.assert_equal(centered.values, self.cell.values)

    def test_isolated2_0(self):
        gap = 1e-10
        iso = self.cell2.isolated2(gap)
        testing.assert_allclose(iso.distances(), self.cell2.distances())

    def test_isolated2_1(self):
        gap = 1e-10
        iso = self.cell.isolated2(gap)
        testing.assert_allclose(iso.distances(), self.cell.distances())

    def test_stack_0(self):
        st = self.cell.stack(self.cell, vector=2)
        nv = self.cell.vectors.copy()
        nv[2, :] *= 2
        testing.assert_allclose(st.vectors, nv)
        testing.assert_allclose(st.coordinates,
                                ((0., 0., 0.), (1. / 3., 1. / 3., 0.25), (0., 0., .5), (1. / 3., 1. / 3., 0.75)))
        testing.assert_equal(st.values, ('Co',) * 4)

    def test_stack_1(self):
        st = self.cell2.stack(self.cell2, vector=0)
        nv = self.cell2.vectors.copy()
        nv[0, :] *= 2
        testing.assert_allclose(st.vectors, nv)
        testing.assert_allclose(st.coordinates,
                                ((0., 0., 0.), (.5, 0., 0.)))
        testing.assert_equal(st.values, ('Co',) * 2)

    def test_stack_2(self):
        st = self.cell2.stack(self.empty, self.cell2, vector=0)
        nv = self.cell2.vectors.copy()
        nv[0, :] *= 3
        testing.assert_allclose(st.vectors, nv)
        testing.assert_allclose(st.coordinates,
                                ((0., 0., 0.), (2. / 3, 0., 0.),))
        testing.assert_equal(st.values, ('Co', 'Co'))

    def test_stack_fail_0(self):
        with self.assertRaises(ValueError):
            self.cell.stack(self.cell2, vector=2, restrict_collinear=True)

    def test_repeated_0(self):
        rp = self.cell.repeated(1, 1, 1)
        testing.assert_allclose(rp.vectors, self.cell.vectors)
        testing.assert_allclose(rp.coordinates, self.cell.coordinates)
        testing.assert_equal(rp.values, self.cell.values)

    def test_repeated_1(self):
        rp = self.cell.repeated(2, 2, 1)
        nv = self.cell.vectors.copy()
        nv[0:2, :] *= 2
        testing.assert_allclose(rp.vectors, nv)
        testing.assert_allclose(rp.coordinates, ((0., 0., 0.),
                                                 (1. / 6, 1. / 6, 0.5), (.5, 0., 0.), (2. / 3, 1. / 6, 0.5),
                                                 (0., 0.5, 0.), (1. / 6, 2. / 3, 0.5), (0.5, 0.5, 0),
                                                 (2. / 3., 2. / 3, 0.5)), atol=1e-14)
        testing.assert_equal(rp.values, ("Co",) * 8)

    def test_select_0(self):
        testing.assert_equal(self.cell.select((0., 0., 0., 1., 1., 1.)), (True, True))

    def test_select_1(self):
        testing.assert_equal(self.cell.select((0., 0., 0., .1, .1, .1)), (True, False))

    def test_select_2(self):
        testing.assert_equal(self.cell2.select((0.1, 0.1, 0.1, 1., 1., 1.)), (False,))

    def test_select_fail(self):
        with self.assertRaises(ValueError):
            self.cell2.select(0, 0, 1, 1)
        with self.assertRaises(ValueError):
            self.cell2.select(0, 0, 1, 1, 1)

    def test_apply(self):
        c = self.cell.apply((False, True))
        testing.assert_equal(c.vectors, self.cell.vectors)
        testing.assert_allclose(c.coordinates, ((1. / 3., 1. / 3., 0.5),))
        testing.assert_equal(c.values, ('Co',))

    def test_discard_0(self):
        c = self.cell.discard(self.cell.select(1. / 6, 1. / 6, 0., 1., 1., 1.))
        testing.assert_equal(c.vectors, self.cell.vectors)
        testing.assert_allclose(c.coordinates, ((0., 0., 0.),))
        testing.assert_equal(c.values, ('Co',))

    def test_discard_1(self):
        c = self.cell.discard((False, True))
        testing.assert_equal(c.vectors, self.cell.vectors)
        testing.assert_allclose(c.coordinates, ((0., 0., 0.),))
        testing.assert_equal(c.values, ('Co',))

    def test_normalized(self):
        c = self.cell.copy(coordinates=self.cell.coordinates + 0.6, values=['Co', 'O'])

        c1 = c.normalized()
        testing.assert_equal(c1.vectors, self.cell.vectors)
        testing.assert_allclose(c1.coordinates, ((.6, .6, .6), (1. / 3 + .6, 1. / 3 + .6, 0.1)))
        testing.assert_equal(c1.values, ('Co', 'O'))

        c2 = c.normalized(sort='z')
        testing.assert_equal(c2.vectors, self.cell.vectors)
        testing.assert_allclose(c2.coordinates, ((1. / 3 + .6, 1. / 3 + .6, 0.1), (.6, .6, .6)))
        testing.assert_equal(c2.values, ('O', 'Co'))

        assert c.normalized(sort=1) == c1
        assert c.normalized(sort=(1, 1, 1)) == c2
        assert c.normalized(sort=(-1, -1, -1)) == c1

    def test_packed(self):
        cell = Cell(
            Basis(self.cell.vectors),
            coordinates=((0.1, 0.1, 0), (0.9, 0.1, 0), (0.1, 0.9, 0), (0.9, 0.9, 0)),
            values=["A"] * 4,
        ).ws_packed()
        testing.assert_allclose(cell.coordinates, (
            (0.1, 0.1, 0),
            (-0.1, 0.1, 0),
            (0.1, -0.1, 0),
            (-0.1, -0.1, 0)
        ))

    def test_isolated_0(self):
        c = self.cell.isolated(1, 2, 3)
        nv = self.cell.vectors.copy()
        nv[0, :] *= 2
        nv[1, :] *= 3
        nv[2, :] *= 4
        testing.assert_allclose(c.vectors, nv)
        testing.assert_allclose(c.coordinates, ((1. / 4, 1. / 3, 3. / 8), (5. / 12, 4. / 9, 1. / 2)))
        testing.assert_equal(c.values, ('Co',) * 2)

    def test_isolated_1(self):
        c = self.cell.isolated(self.a, 2 * self.a, 3 * self.h, units='cartesian')
        nv = self.cell.vectors.copy()
        nv[0, :] *= 2
        nv[1, :] *= 3
        nv[2, :] *= 4
        testing.assert_allclose(c.vectors, nv)
        testing.assert_allclose(c.coordinates, ((1. / 4, 1. / 3, 3. / 8), (5. / 12, 4. / 9, 1. / 2)))
        testing.assert_equal(c.values, ('Co',) * 2)

    def test_isolated_fail(self):
        with self.assertRaises(ValueError):
            self.cell.isolated(1, 2, 3, units='unknown')

    def test_add(self):
        c = self.cell.copy(coordinates=.95 - self.cell.coordinates)
        s = self.cell.merge(c)
        testing.assert_allclose(s.coordinates, (
        (0., 0., 0.), (1. / 3, 1. / 3, .5), (.95, .95, .95), (.95 - 1. / 3, .95 - 1. / 3, .45)))
        testing.assert_equal(s.values, ('Co',) * 4)

    def test_add_fail(self):
        with self.assertRaises(ValueError):
            self.cell.merge(self.cell2)

    def test_species_0(self):
        sp = self.cell.species()
        assert len(sp) == 1
        assert sp['Co'] == 2

    def test_species_1(self):
        c = self.cell.copy(values=['C', 'Co'])
        c = c.merge(self.cell)
        sp = c.species()
        assert len(sp) == 2
        assert sp['Co'] == 3
        assert sp['C'] == 1

    def test_reorder_0(self):
        c = self.cell.transpose_vectors(0, 2, 1)
        testing.assert_equal(c.vectors, self.cell.vectors[(0, 2, 1), ...])
        testing.assert_allclose(c.coordinates, ((0., 0., 0.), (1. / 3, .5, 1. / 3)))
        testing.assert_equal(c.values, ('Co', 'Co'))

    def test_pickle(self):
        c = pickle.loads(pickle.dumps(self.cell))
        testing.assert_equal(c.vectors, self.cell.vectors)
        testing.assert_equal(c.coordinates, self.cell.coordinates)
        testing.assert_equal(c.values, self.cell.values)

    def test_supercell(self):
        s = self.cell.supercell(
            (1, 0, 0),
            (-1, 2, 0),
            (0, 0, 1)
        )
        s = s.normalized(sort='y')

        testing.assert_allclose(s.vectors, (
            (self.a, 0, 0),
            (0, self.a * 3. ** .5, 0),
            (0, 0, self.h)
        ))
        testing.assert_allclose(s.coordinates, (
            (0, 0, 0.),
            (0.5, 1. / 6, .5),
            (0.5, 0.5, 0.),
            (0, 2. / 3, .5),
        ), atol=1e-10)
        testing.assert_equal(s.values, ('Co',) * 4)

    def test_interpolate(self):
        c = Cell(
            Basis.orthorhombic((1, 1), meta={'key': 'value'}),
            (
                (.5, .5),
                (0, 0),
                (0, .5),
                (.5, 0),
            ),
            ((1, 5), (2, 6), (3, 7), (4, 8))
        )

        for p in (True, False):
            c2 = c.interpolate((0, 0), (.5, .5), periodic=p)

            testing.assert_equal(c.vectors, c2.vectors)
            testing.assert_equal(c.meta, c2.meta)
            testing.assert_equal(c2.coordinates, ((0, 0), (.5, .5)))
            testing.assert_allclose(c2.values, ((2, 6), (1, 5)))

    def test_serialization(self):
        serialized = self.cell.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.cell.vectors.tolist(),
            meta={},
            type="pycoordinates.cell.Cell",
            coordinates=self.cell.coordinates.tolist(),
            values=self.cell.values.tolist(),
        ))
        assert self.cell == Cell.from_state_dict(serialized)
        with self.assertRaises(TypeError):
            Basis.from_state_dict(serialized, check_type=True)


class FCCCellTest(TestCase):

    def test_sc_roundoff(self):
        si_basis = Basis.triclinic((3.9 / 2, 3.9 / 2, 3.9 / 2), (.5, .5, .5))
        si_cell = Cell(si_basis, (.5, .5, .5), 'Si')
        cubic_cell = si_cell.supercell(
            (1, -1, 1),
            (1, 1, -1),
            (-1, 1, 1),
        )
        testing.assert_allclose(
            cubic_cell.size / cubic_cell.volume,
            si_cell.size / si_cell.volume,
        )
