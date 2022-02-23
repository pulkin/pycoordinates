from .util import generate_path, uniform_grid, roarray_copy, orthogonal_basis
from .basis import Basis

import numpy as np
from numpy import testing
import pytest


def test_generate_path_0():
    keys = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    pth = generate_path(keys, 7)
    testing.assert_allclose(pth, (
        (0, 0, 0),
        (.5, 0, 0),
        (1, 0, 0),
        (1, .5, 0),
        (1, 1, 0),
        (1, 1, .5),
        (1, 1, 1)
    ))


def test_generate_path_1():
    keys = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    pth = generate_path(keys, 7)
    for k in keys:
        a = np.array(k, dtype=float).tolist()
        b = pth.tolist()
        assert a in b


def test_generate_path_2():
    keys = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    pth = generate_path(keys, 6, skip_segments=(1,))
    testing.assert_allclose(pth, (
        (0, 0, 0),
        (.5, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, .5),
        (1, 1, 1)
    ))

    pth = generate_path(keys, 5, skip_segments=(0,))
    testing.assert_allclose(pth, (
        (1, 0, 0),
        (1, .5, 0),
        (1, 1, 0),
        (1, 1, .5),
        (1, 1, 1)
    ))

    pth = generate_path(keys, 5, skip_segments=(2,))
    testing.assert_allclose(pth, (
        (0, 0, 0),
        (.5, 0, 0),
        (1, 0, 0),
        (1, .5, 0),
        (1, 1, 0),
    ))

    pth = generate_path(keys, 3, skip_segments=(1, 0))
    testing.assert_allclose(pth, (
        (1, 1, 0),
        (1, 1, .5),
        (1, 1, 1)
    ))

    pth = generate_path(keys, 3, skip_segments=(2, 0))
    testing.assert_allclose(pth, (
        (1, 0, 0),
        (1, .5, 0),
        (1, 1, 0),
    ))

    with pytest.raises(ValueError):
        generate_path(keys, 3, skip_segments=(1,))

    with pytest.raises(ValueError):
        generate_path(keys, 3, skip_segments=(0, 1, 2))


def test_generate_path_3():
    c = Basis.triclinic((1, 1, .1), (0, 0, .5))
    keys = c.transform_to_cartesian(np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1)]))
    pth = c.transform_from_cartesian(generate_path(keys, 6))
    testing.assert_allclose(pth, (
        (0, 0, 0),
        (.25, 0, 0),
        (.5, 0, 0),
        (.75, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
    ))


def test_uniform_grid():
    c = uniform_grid((1, 2, 3))
    testing.assert_equal(c, [[
        [
            (0, 0, 0),
            (0, 0, 1. / 3),
            (0, 0, 2. / 3),
        ], [
            (0, .5, 0),
            (0, .5, 1. / 3),
            (0, .5, 2. / 3),
        ],
    ]])


def test_array_copy_subclass():
    class myarray(np.ndarray):
        pass
    test = myarray((3,))
    assert type(test) is myarray
    test_copy = roarray_copy(test)
    testing.assert_equal(test, test_copy)
    assert type(test_copy) is myarray


def test_orthogonal_basis():
    b = orthogonal_basis([
        (0, 0, 1),
        (1, 0, 0.5),
    ])
    testing.assert_allclose(b[:2], [
        (0, 0, 1),
        (1, 0, 0),
    ])
    testing.assert_allclose(np.abs(b[2:]), [
        (0, 1, 0),
    ])
