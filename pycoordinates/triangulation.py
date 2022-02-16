import numpy as np
from numpy import ndarray
from collections import namedtuple
from scipy.special import factorial

from .util import roarray
from .tetrahedron2 import compute_density_from_triangulation


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
    ], (2, 2, 2)), (1, 2, 0)).astype(np.int32)),  # [6, 4, 3]
    2: roarray(np.transpose(np.unravel_index([
        (0, 1, 2),
        (2, 1, 3),
    ], (2, 2)), (1, 2, 0)).astype(np.int32)),
}


def unique_counts(a: ndarray) -> ndarray:
    """
    Counts unique elements in an [N, n] array along the last axis.

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


def compute_band_density(triangulation: triangulation_result, values: ndarray, points: ndarray,
                         weights: ndarray = None, resolve_bands: bool = False) -> ndarray:
    """
    Computes band density.
    3D only.

    Parameters
    ----------
    triangulation
        Triangulation to use.
    values
        Band values.
    points
        Values to compute density at.
    weights
        Optional weights to multiply densities.
    resolve_bands
        If True, resolves bands.

    Returns
    -------
    densities
        The resulting densities.
    """
    assert triangulation.simplices.shape[1] == 4, "Triangulation is not tetrahedrons"
    simplices_here = triangulation.points_i[triangulation.simplices]
    if weights is not None:
        weights = weights.reshape(values.shape)
    return compute_density_from_triangulation(
        simplices_here, triangulation.weights, values, points,
        band_weights=weights,
        resolve_bands=resolve_bands)
