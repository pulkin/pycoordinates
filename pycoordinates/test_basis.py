from .basis import Basis

import numpy as np
from numpy import testing
import pickle
import pytest


def test_basis_init_0():
    b = Basis(((1, 0), (0, 1)))
    testing.assert_equal(b.vectors, ((1, 0), (0, 1)))
    assert b.meta == dict()


def test_basis_init_1():
    b = Basis(np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9))))
    testing.assert_equal(b.vectors, np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9))))


def test_basis_init_2():
    b = Basis.orthorhombic((1, 2, 3))
    testing.assert_equal(b.vectors, np.array(((1, 0, 0), (0, 2, 0), (0, 0, 3))))


def test_basis_init_3():
    b = Basis.triclinic((1, 1, 3), (0, 0, 0.5))
    testing.assert_allclose(b.vectors, np.array(((1, 0, 0), (0.5, 3. ** .5 / 2, 0), (0, 0, 3))), atol=1e-7)


def test_basis_init_4():
    b = Basis(np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9))))
    c = Basis(b)
    testing.assert_equal(b.vectors, c.vectors)


def test_basis_init_5():
    a = np.array(((1, 0), (0, 1)))
    meta = {"a": "b"}
    b = Basis(a, meta=meta)
    assert b.vectors is not a
    assert b.meta is not meta
    assert b.meta == meta

    c = Basis(b)
    assert c.meta == {}

    c = Basis(b, meta={"c": "d"})
    assert c.meta == {"c": "d"}


def test_basis_init_6():
    testing.assert_allclose(Basis.diamond(3.14).volume, 3.14 ** 3 / 4)


class TestBasis:
    def setup_method(self):
        self.b = Basis.triclinic((1, 1, 3), (0, 0, 0.5), meta={"key": "value"})
        self.c = Basis.orthorhombic((1, 1, 1))

    def test_pickle(self):
        b = pickle.loads(pickle.dumps(self.b))
        testing.assert_equal(b.vectors, self.b.vectors)
        testing.assert_equal(b.meta, self.b.meta)

    def test_eq(self):
        assert self.b == Basis(self.b)
        assert not self.b == self.c

    def test_round(self):
        b = self.b.rounded(1)
        testing.assert_allclose(b.vectors, [(1, 0, 0), (.5, .9, 0), (0, 0, 3)])

    def test_transform(self):
        coordinates = np.array((1, 1, 1))

        transformed = self.b.transform_to(self.c, coordinates)
        testing.assert_allclose(transformed, (1.5, 3. ** .5 / 2, 3), atol=1e-7)

        transformed = self.b.transform_from(self.c, transformed)
        testing.assert_allclose(transformed, coordinates, atol=1e-7)

        transformed = self.b.transform_to_cartesian(coordinates)
        testing.assert_allclose(transformed, (1.5, 3. ** .5 / 2, 3), atol=1e-7)

        transformed = self.b.transform_from_cartesian(transformed)
        testing.assert_allclose(transformed, coordinates, atol=1e-7)

        transformed = self.b.transform_to(self.c, coordinates[None, :])
        testing.assert_allclose(transformed, ((1.5, 3. ** .5 / 2, 3),), atol=1e-7)

    def test_volume(self):
        assert abs(self.b.volume - 1.5 * 3. ** .5) < 1e-7

    def test_reciprocal(self):
        testing.assert_allclose(self.b.reciprocal.vectors.T @ self.b.vectors, np.eye(3), atol=1e-10)

    def test_vertices(self):
        v = self.b.vertices
        s = 0.5 * 3. ** .5
        testing.assert_allclose(v, (
            (0, 0, 0),
            (0, 0, 3),
            (.5, s, 0),
            (.5, s, 3),
            (1, 0, 0),
            (1, 0, 3),
            (1.5, s, 0),
            (1.5, s, 3),
        ), atol=1e-7)

    def test_edges(self):
        v = self.b.edges
        s = 0.5 * 3. ** .5
        testing.assert_allclose(v, (
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 3), (1, 0, 3)),
            ((.5, s, 0), (1.5, s, 0)),
            ((.5, s, 3), (1.5, s, 3)),
            ((0, 0, 0), (.5, s, 0)),
            ((0, 0, 3), (.5, s, 3)),
            ((1, 0, 0), (1.5, s, 0)),
            ((1, 0, 3), (1.5, s, 3)),
            ((0, 0, 0), (0, 0, 3)),
            ((.5, s, 0), (.5, s, 3)),
            ((1, 0, 0), (1, 0, 3)),
            ((1.5, s, 0), (1.5, s, 3)),
        ), atol=1e-7)

    def test_copy(self):
        b2 = self.b.copy()
        testing.assert_equal(self.b.vectors, b2.vectors)
        testing.assert_equal(self.b.meta, b2.meta)
        assert self.b.vectors is not b2.vectors
        assert self.b.meta is not b2.meta

    def test_stack(self):
        nv = self.b.vectors.copy()

        st = self.b.stack(self.b, vector='y')
        nv[1] *= 2
        testing.assert_allclose(st.vectors, nv)

        st = st.stack(st, st, vector='x')
        nv[0] *= 3
        testing.assert_allclose(st.vectors, nv)

    def test_stack_raises(self):
        with pytest.raises(ValueError):
            self.b.stack(self.c, vector=0)

    def test_repeated(self):
        nv = self.b.vectors.copy()

        r = self.b.repeated(3, 2, 1)
        nv[0] *= 3
        nv[1] *= 2
        testing.assert_allclose(r.vectors, nv)

    def test_repeated_fail(self):
        with pytest.raises(TypeError):
            self.b.repeated(2.0, 2.0, 1.0)

    def test_rotated_0(self):
        b1 = self.b.rotated(np.array((0, 0, -1)), np.pi / 2)
        testing.assert_allclose(b1.vectors, (
            (0, 1, 0),
            (-3. ** .5 / 2, .5, 0),
            (0, 0, 3),
        ), atol=1e-7)

    def test_rotated_1(self):
        c1 = self.c.rotated(np.array((0, 0, -1)), np.pi / 4)
        s2 = 1. / 2. ** .5
        testing.assert_allclose(c1.vectors, (
            (s2, s2, 0),
            (-s2, s2, 0),
            (0, 0, 1),
        ))

    def test_serialization(self):
        serialized = self.b.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.b.vectors.tolist(),
            meta=dict(key="value"),
            type="pycoordinates.basis.Basis"
        ))
        assert self.b == Basis.from_state_dict(serialized)
