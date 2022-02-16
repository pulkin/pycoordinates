import numpy as np
from numpy import ndarray
from collections import namedtuple
from scipy.special import factorial
from attr import attrs, attrib

from .util import roarray
from .attrs import convert_tri_points, convert_tri_points_i, convert_tri_weights, convert_tri_simplices,\
    check_tri_points, check_tri_points_i, check_tri_weights, check_tri_simplices
from .tetrahedron2 import compute_density_from_triangulation


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


@attrs(frozen=True, eq=False)
class Triangulation:
    """Describes triangulation of multiple points."""
    points = attrib(type=ndarray, converter=convert_tri_points, validator=check_tri_points)
    points_i = attrib(type=ndarray, converter=convert_tri_points_i, validator=check_tri_points_i)
    simplices = attrib(type=ndarray, converter=convert_tri_simplices, validator=check_tri_simplices)
    weights = attrib(type=ndarray, converter=convert_tri_weights, validator=check_tri_weights)

    def compute_band_density(self, values: ndarray, points: ndarray, weights: ndarray = None,
                             resolve_bands: bool = False) -> ndarray:
        """
        Computes band density.
        3D only.

        Parameters
        ----------
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
        assert self.simplices.shape[1] == 4, "Triangulation is not tetrahedrons"
        simplices_here = self.points_i[self.simplices]
        if weights is not None:
            weights = weights.reshape(values.shape)
        return compute_density_from_triangulation(
            simplices_here, self.weights, values,
            points,
            band_weights=weights,
            resolve_bands=resolve_bands)


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
