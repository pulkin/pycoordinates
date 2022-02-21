# cython: language_level=3
import numpy
cimport numpy, cython


@cython.cdivision(True)
cdef double _density(double e, double e1, double e2, double e3, double e4):
    if e1 < e <= e2:
        return 3 * (e - e1) * (e - e1) / (e2 - e1) / (e3 - e1) / (e4 - e1)
    elif e2 < e <= e3:
        return (3 * (e2 - e1) + 6 * (e - e2) - 3 * (e4 + e3 - e2 - e1) * (e - e2) * (e - e2) / (e3 - e2) / (e4 - e2)) / (e3 - e1) / (e4 - e1)
    elif e3 < e < e4:
        return 3 * (e4 - e) * (e4 - e) / (e4 - e1) / (e4 - e2) / (e4 - e3)
    else:
        return 0


cdef inline void sort2(double* _p0, double* _p1):
    cdef double p0 = _p0[0]
    cdef double p1 = _p1[0]
    cdef double t
    if p0 > p1:
        t = _p0[0]
        _p0[0] = _p1[0]
        _p1[0] = t


cdef inline void sort4(double* p0, double* p1, double* p2, double* p3):
    sort2(p0, p1)
    sort2(p2, p3)
    sort2(p0, p2)
    sort2(p1, p3)
    sort2(p1, p2)


@cython.wraparound(False)
def compute_density_from_triangulation(
        const int[:, ::1] triangulation,  # [n_tri, 4]
        const double[::1] tri_weights,  # [n_tri]
        const double[:, ::1] bands,  # [n_pts, n_bands]
        const double[::1] target,  # [n_target]
        object band_weights = None,  # [n_pts, n_bands]
        int resolve_bands = False,
):  # [n_tri, n_bands, n_target] or [n_tri, n_target]
    """
    Based on previously computed triangulation, evaluates tetrahedron band density.

    Parameters
    ----------
    triangulation
        A 2D `[n_tri, 4]` array with simplex (tetrahedron) indices.
    tri_weights
        A 1D `[n_tri]` array with simplex weights coming as a pre-factor
        for each density point.
    bands
        A 2D `[n_pts, n_bands]` array with band energies for the density.
    target
        A 1D `[n_target]` array with target energies where the density has
        to be computed.
    band_weights
        An optional 2D `[n_pts, n_bands]` array with band weights. While
        `tri_weights` are defined per tetrahedron, band weights are
        defined per-vertex per-band.
    resolve_bands
        If True, resolves the density per-band.

    Returns
    -------
    result
        Depending on `resolve_bands`, a 2D `[n_tri, n_target]`
        or a 3D `[n_tri, n_bands, n_target]` array with densities.
    """
    assert triangulation.shape[1] == 4

    cdef int n_tri = triangulation.shape[0]
    cdef int n_bands = bands.shape[1]
    cdef int n_target = target.shape[0]

    cdef int t1, t2, t3, t4, trix, bix, i
    cdef double e1, e2, e3, e4, w, d

    cdef double[:, ::1] band_weights_buffer = band_weights

    cdef double[:, :, ::1] result_resolved
    cdef double[:, ::1] result
    if resolve_bands:
        result_resolved = numpy.zeros((n_tri, n_bands, n_target))
    else:
        result = numpy.zeros((n_tri, n_target))

    for trix in range(n_tri):
        t1 = triangulation[trix, 0]
        t2 = triangulation[trix, 1]
        t3 = triangulation[trix, 2]
        t4 = triangulation[trix, 3]
        w = tri_weights[trix]

        for bix in range(n_bands):
            e1 = bands[t1, bix]
            e2 = bands[t2, bix]
            e3 = bands[t3, bix]
            e4 = bands[t4, bix]
            sort4(&e1, &e2, &e3, &e4)

            if band_weights_buffer != None:
                w *= (band_weights_buffer[t1, bix] +
                      band_weights_buffer[t2, bix] +
                      band_weights_buffer[t3, bix] +
                      band_weights_buffer[t4, bix]) / 4

            for i in range(n_target):
                d = _density(target[i], e1, e2, e3, e4) * w
                if resolve_bands:
                    result_resolved[trix, bix, i] = d
                else:
                    result[trix, i] += d
    if resolve_bands:
        return numpy.asarray(result_resolved)
    else:
        return numpy.asarray(result)
