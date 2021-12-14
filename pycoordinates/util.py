from numpy import array, ndarray
import numpy as np

from functools import partial
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
