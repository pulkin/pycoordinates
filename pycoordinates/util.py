from numpy import array, ndarray
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from functools import partial, wraps
from typing import Union


def roarray(a: ndarray) -> ndarray:
    """Flags numpy array as read-only."""
    a.flags.writeable = False
    return a


def roarray_copy(a: ndarray, **kwargs) -> ndarray:
    """Read-only copy."""
    return roarray(array(a, **kwargs))


ro_float_array_copy = partial(roarray_copy, dtype=float)


def compute_angles(v1: ndarray, v2: ndarray, axis: int = -1) -> ndarray:
    """
    Computes angle cosines between sets of vectors.

    Parameters
    ----------
    v1
    v2
        Sets of vectors.
    axis
        Dimension to sum over.

    Returns
    -------
    A numpy array with angle cosines.
    """
    return (v1 * v2).sum(axis=axis) / ((v1 ** 2).sum(axis=axis) * (v2 ** 2).sum(axis=axis)) ** .5


def generate_path(nodes: Union[list, tuple, ndarray], n: int, skip_segments: Union[list, tuple, ndarray] = None) -> ndarray:
    """
    Distributes ``n`` points uniformly along a path specified by nodes.

    Parameters
    ----------
    nodes
        A list or a 2D array of nodes' coordinates.
    n
        The desired point count in the path.
    skip_segments
        An optional array with segment indices to skip.

    Returns
    -------
    The resulting path.
    """
    def interpolate(_p1: ndarray, _p2: ndarray, _n, _e):
        x = np.linspace(0, 1, _n + 2)[:, None]
        if not _e:
            x = x[:-1]
        return _p1[None, :] * (1 - x) + _p2[None, :] * x

    if skip_segments is None:
        skip_segments = tuple()
    skip_segments = np.array(skip_segments, dtype=int)

    nodes = np.asanyarray(nodes)
    lengths = np.linalg.norm(nodes[:-1] - nodes[1:], axis=1)

    mask_segment = np.ones(len(nodes), dtype=bool)
    mask_segment[skip_segments] = False
    mask_segment[-1] = False
    n_reserved = (np.logical_or(mask_segment[1:], mask_segment[:-1]).sum())
    n_reserved += mask_segment[0]

    if n_reserved == 0:
        raise ValueError("Empty edges specified")

    if n < n_reserved:
        raise ValueError("The number of points is less then the number of edges {:d} < {:d}".format(n, n_reserved))

    mask_endpoint = np.logical_not(mask_segment[1:])
    mask_segment = mask_segment[:-1]

    points_l = nodes[:-1][mask_segment]
    points_r = nodes[1:][mask_segment]
    lengths = lengths[mask_segment]
    buckets = np.zeros(len(lengths), dtype=int)
    endpoints = mask_endpoint[mask_segment]
    for i in range(n - n_reserved):
        dl = lengths / (buckets + 1)
        buckets[np.argmax(dl)] += 1
    result = []
    for pt1, pt2, _n, e in zip(points_l, points_r, buckets, endpoints):
        result.append(interpolate(pt1, pt2, _n, e))
    return np.concatenate(result)


def grid_coordinates(arrays: tuple) -> np.ndarray:
    """
    Transforms several 1D arrays of coordinates into
    (N + 1)D mesh coordinate array.

    Parameters
    ----------
    arrays
        1D arrays as a tuple.

    Returns
    -------
    A multidimensional coordinate array.
    """
    mg = np.meshgrid(*arrays, indexing='ij')
    return np.concatenate(tuple(i[..., None] for i in mg), axis=len(mg))


def uniform_grid(size: tuple, endpoint: bool = False) -> np.ndarray:
    """
    Sample a multidimensional 0-1 interval box.

    Parameters
    ----------
    size
        A tuple of integers specifying point count per dimension.
    endpoint
        Whether to include 1 to the right of the interval.

    Returns
    -------
    A multidimensional coordinate array.
    """
    return grid_coordinates(tuple(
        np.linspace(0, 1, i, endpoint=endpoint)
        for i in size
    ))  # rewrite with np.mgrid?


def ravel_grid(a: ndarray, shape: tuple) -> ndarray:
    """
    Ravels grid coordinates.

    Parameters
    ----------
    a
        Grid coordinates.
    shape
        Grid dimensions.

    Returns
    -------
    result
        A tensor with the last dimension squeezed into a big index.
    """
    return np.ravel_multi_index(tuple(
        a[..., i]
        for i in range(a.shape[-1])
    ), shape)


def input_as_list(func):
    """Transforms multiple position-only arguments into a list."""
    @wraps(func)
    def a_w(*args, **kwargs):
        self = args[0]
        if len(args) > 2:
            args = [self, list(args[1:])]
        elif len(args) == 2:
            try:
                iter(args[1])
            except TypeError:
                args = [self, list(args[1:])]
        else:
            args = [self, []]

        return func(*args, **kwargs)

    return a_w


def qhull_interpolation_driver(points: ndarray, values: ndarray, points_i: ndarray, **kwargs) -> ndarray:
    """
    Delaunay interpolation driver provided by Qhull and interfaced with scipy.

    Parameters
    ----------
    points : ndarray
        Sparse point coordinates.
    values : ndarray
        Point values.
    points_i : ndarray
        Target point coordinates to interpolate at.
    kwargs
        Arguments to constructor.

    Returns
    -------
    The interpolated values.
    """
    return LinearNDInterpolator(points, values, **kwargs)(points_i)


def derived_from(derived: ndarray, reference: ndarray) -> ndarray:
    """
    Arrays derived from other arrays.
    Used to cast units between arrays (stub for now).

    Parameters
    ----------
    derived : ndarray
        Derived array.
    reference : ndarray
        The reference array.

    Returns
    -------
    The resulting array adjusted.
    """
    return derived


def _piece2bounds(piece: Union[ndarray, list, tuple], dim: int) -> (ndarray, ndarray):
    piece = np.reshape(piece, (2, dim))
    return np.amin(piece, axis=0), np.amax(piece, axis=0)
