from .cell import Cell

from numpy import testing
import numpy as np
from unittest import TestCase


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
