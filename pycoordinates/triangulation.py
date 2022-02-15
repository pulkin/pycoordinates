import numpy as np
from numpy import ndarray
from collections import namedtuple
from scipy.special import factorial

from .util import roarray


triangulation_result = namedtuple("triangulation_result", ("points", "points_i", "simplices", "weights"))
nan = float("nan")
_lookup_unique_counts = {
    4: roarray(np.array([
        nan, nan, nan, nan, 4,
        nan, 3,
        nan, 2,
        nan, 2,
        nan, nan, nan, nan, nan, 1,
    ])),
    3: roarray(np.array([
        nan, nan, nan, 3,
        nan, 2,
        nan, nan, nan, 1,
    ]))
}
cube_tetrahedrons = {
    3: roarray(np.transpose(np.unravel_index([
        (0, 1, 2, 5),
        (1, 2, 3, 5),
        (0, 2, 4, 5),
        (2, 4, 5, 6),
        (2, 3, 5, 7),
        (2, 5, 6, 7),
    ], (2, 2, 2)), (1, 2, 0))),  # [6, 4, 3]
    2: roarray(np.transpose(np.unravel_index([
        (0, 1, 2),
        (2, 1, 3),
    ], (2, 2)), (1, 2, 0))),
}


def unique_counts(a: ndarray) -> ndarray:
    """
    Counts unique elements in an [N, 4] array.

    Parameters
    ----------
    a
        The array to process.

    Returns
    -------
    result
        The resulting unique counts.
    """

    return _lookup_unique_counts[a.shape[-1]][
        (a[..., :, None] == a[..., None, :]).sum(axis=(-1, -2))
    ]


def simplex_volumes(a: ndarray) -> ndarray:
    """
    Computes simplex volumes.

    Parameters
    ----------
    a
        Array with cartesian coordinates.

    Returns
    -------
    result
        The resulting volumes.
    """
    assert a.shape[-1] == a.shape[-2] - 1
    n = a.shape[-1]
    return np.abs(np.linalg.det(a[..., :-1, :] - a[..., -1:, :])) / factorial(n)

