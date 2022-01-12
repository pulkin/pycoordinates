from __future__ import annotations

from .basis import Basis, _gaps2x
from . import cell
from .util import input_as_list, derived_from, grid_coordinates, generate_path, _piece2bounds
from .attrs import check_vectors_inv, convert_vectors_inv, convert_grid, check_grid, convert_grid_values,\
    check_grid_values
from .tetrahedron import compute_density, compute_density_resolved

import numpy as np
from numpy import ndarray
from attr import attrs, attrib

from typing import Union
from functools import cached_property


@attrs(frozen=True, eq=False)
class Grid(Basis):
    """Describes data on a grid."""
    coordinates = attrib(type=tuple, converter=convert_grid, validator=check_grid)
    values = attrib(type=Union[ndarray, list, tuple, str], converter=convert_grid_values, validator=check_grid_values)
    meta = attrib(type=dict, factory=dict, converter=dict)
    _vectors_inv = attrib(type=Union[ndarray, list, tuple], default=None, converter=convert_vectors_inv,
                          validator=check_vectors_inv)

    @cached_property
    def grid_shape(self) -> tuple:
        return tuple(map(len, self.coordinates))

    @cached_property
    def size(self) -> int:
        return int(np.prod(self.grid_shape))

    def __eq__(self, other):
        return super().__eq__(other) and all(np.array_equal(*i) for i in zip(self.coordinates, other.coordinates)) and \
               np.array_equal(self.values, other.values)

    def explicit_coordinates(self) -> np.ndarray:  # TODO: rename to get_grid_coordinates
        """
        Prepares an (N + 1)D array with grid coordinates.

        Returns
        -------
        A multidimensional coordinate array.
        """
        return grid_coordinates(self.coordinates)

    def cartesian(self) -> np.ndarray:  # TODO: rename to get_grid_cartesian
        """
        Computes cartesian coordinates of points in this grid.

        Returns
        -------
        A multidimensional coordinate array.
        """
        return self.transform_to_cartesian(self.explicit_coordinates())

    def normalized(self, left: float = 0, sort: bool = False) -> Grid:
        """
        Puts all grid points inside box boundaries and returns a copy.

        Parameters
        ----------
        left : float
            The left edge of the normalized box in cell
            coordinates. For example, ``left=-0.3`` stands
            for coordinates being placed in a ``[-0.3, 0.7)``
            interval.
        sort : bool
            Optionally, sort all grid point coordinates in ascending
            order.

        Returns
        -------
        A new grid with the normalized data.
        """
        d = self.state_dict(mark_type=False)
        d["coordinates"] = new_coordinates = list(((i - left) % 1) + left for i in d["coordinates"])
        if sort:
            values = d["values"]
            for dim, c in enumerate(new_coordinates):
                order = np.argsort(c)
                new_coordinates[dim] = c[order]
                values = values[(slice(None),) * dim + (order,)]
            d["values"] = values
        return self.__class__(**d)

    @input_as_list
    def isolated(self, gaps: list, units: str = "cartesian") -> Grid:
        """
        Isolates points from their images in this cell or grid by elongating basis
        vectors while keeping distances between the points fixed.

        Parameters
        ----------
        gaps : list
            The elongation amount in cartesian or in crystal units.
        units : str
            Units of `gaps`: 'cartesian' or 'crystal'.

        Returns
        -------
        A bigger cell where points are spatially isolated from their images.
        """
        gaps = _gaps2x(self, gaps, units)
        vectors = self.vectors * gaps[..., None]
        coordinates = [(c + 0.5 * (g - 1)) / g for c, g in zip(self.coordinates, gaps)]
        return self.copy(vectors=vectors, coordinates=coordinates)

    @input_as_list
    def select(self, piece: list) -> list:
        """
        Selects points in this cell or grid inside a box defined in the crystal basis.
        Images are not included.

        Parameters
        ----------
        piece : list
            Box dimensions ``[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]``,
            where x, y, z are basis vectors.

        Returns
        -------
        A numpy array with the selection mask.
        """
        p1, p2 = _piece2bounds(piece, len(self.vectors))
        return list((c < mx) & (c >= mn) for c, mn, mx in zip(self.coordinates, p1, p2))

    @input_as_list
    def apply(self, selection: list) -> Grid:
        """
        Applies a mask to this grid to keep a subset of points.

        Parameters
        ----------
        selection
            A bool mask with selected species.

        Returns
        -------
        The resulting grid.
        """
        selection = list(selection)
        coordinates = []
        values = self.values

        for i, (c, m) in enumerate(zip(self.coordinates, selection)):
            m = np.array(m)
            coordinates.append(c[m])
            values = values[(slice(None),) * i + (m,)]
        return self.copy(coordinates=coordinates, values=values)

    @input_as_list
    def discard(self, selection: list) -> Grid:
        """
        Discards points from this grid according to the mask specified.
        Complements ``self.apply``.

        Parameters
        ----------
        selection
            Points to discard.

        Returns
        -------
        The resulting cell.
        """
        return self.apply(tuple(map(np.logical_not, selection)))

    @input_as_list
    def cut(self, piece: list, selection: Union[ndarray, list, tuple] = None) -> Grid:
        """
        Selects a box inside this grid and returns it in a smaller grid object.
        Basis vectors of the resulting instance are collinear to those of `self`.

        Parameters
        ----------
        piece
            Box dimensions ``[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]``,
            where x, y, z are basis vectors.
        selection
            A custom selection mask or None if all points in the selected box
            have to be included.

        Returns
        -------
        A smaller instance with a subset of points.
        """
        if selection is None:
            selection = self.select(piece)
        p1, p2 = _piece2bounds(piece, len(self.vectors))
        vectors = self.vectors * (p2 - p1)[:, None]
        coordinates = list((c - _p1) / (_p2 - _p1) for c, _p1, _p2 in zip(self.coordinates, p1, p2))
        return self.copy(vectors=vectors, coordinates=coordinates).apply(selection)

    @input_as_list
    def merge(self, grids: list, fill: object = np.nan) -> Grid:
        """
        Merges points from several grids with the same basis.

        Parameters
        ----------
        grids
            Grids to merge.
        fill
            The value to use for missing grid points.

        Returns
        -------
        A new grid with all points merged.
        """
        dims = len(self.coordinates)
        grids = [self] + grids
        new_coordinates = []

        # Coordinates lookup tables
        coord2index = []

        # Calculate unique coordinates on the grid and lookup tables
        for j in range(dims):

            c = []
            for i in grids:
                c.append(i.coordinates[j])

            c = np.concatenate(c, axis=0)
            unique_coordinates, lookup = np.unique(c, return_inverse=True)
            new_coordinates.append(unique_coordinates)
            coord2index.append(lookup)

        new_shape = tuple(a.shape[0] for a in new_coordinates)
        new_values = np.ones(new_shape + self.values.shape[dims:]) * fill

        # Fill in the values
        offsets = [0] * dims
        for i in grids:

            location = tuple(c2i[o:o + c.shape[0]] for o, c2i, c in zip(offsets, coord2index, i.coordinates))
            location = np.ix_(*location)
            new_values[location] = i.values

            for j in range(len(offsets)):
                offsets[j] += i.coordinates[j].shape[0]

        return self.__class__(self, new_coordinates, new_values)

    @input_as_list
    def stack(self, grids: list, vector: int, **kwargs) -> Grid:
        """
        Stack multiple grids along the provided vector.

        Parameters
        ----------
        grids : list
            Grids and bases to stack.
        vector : int
            Basis vector to stack along.
        kwargs
            Other arguments to ``Basis.stack``.

        Returns
        -------
        The resulting grids stacked.
        """
        grids = (self, *grids)
        other_vectors = list(range(grids[0].vectors.shape[0]))
        del other_vectors[vector]
        dims = self.vectors.shape[0]

        basis = Basis.stack(*grids, vector=vector, **kwargs)

        for i, g in enumerate(grids[1:]):
            if isinstance(g, Grid):
                for dim in other_vectors:
                    if not np.all(g.coordinates[dim] == self.coordinates[dim]):
                        raise ValueError(f"Mismatch of grid coordinates along the dimension {dim:d} for cells 0 "
                                         f"and {i:d}")

        values = np.concatenate(tuple(grid.values for grid in grids if isinstance(grid, Grid)), axis=vector)

        stacking_vectors_len = np.asanyarray(tuple((grid.vectors[vector] ** 2).sum(axis=-1) ** .5 for grid in grids))
        shifts = np.cumsum(stacking_vectors_len)
        shifts = shifts / shifts[-1]

        k = np.ones((len(grids), dims))
        k[:, vector] = stacking_vectors_len / stacking_vectors_len.sum()
        b = np.zeros((len(grids), dims))
        b[:, vector] = np.concatenate(((0,), shifts[:-1]))

        coordinates = []
        for dim in range(dims):
            if dim == vector:
                coordinates.append(np.concatenate(tuple(
                    grid.coordinates[dim] * k[i, dim] + b[i, dim] for i, grid in enumerate(grids) if
                    isinstance(grid, Grid)
                ), axis=0))
            else:
                coordinates.append(self.coordinates[dim])

        return self.__class__(basis, coordinates, values)

    @input_as_list
    def transpose_vectors(self, new: list) -> Grid:
        """
        Reorders basis vectors without changing cartesian coordinates.

        Parameters
        ----------
        new
            The new order as a list of integers.

        Returns
        -------
        A new grid with reordered vectors.
        """
        return self.__class__(
            super().transpose_vectors(new),
            tuple(self.coordinates[i] for i in new),
            np.transpose(self.values, new),
            meta=self.meta,
        )

    def rounded(self, decimals: int = 8) -> Grid:
        """
        Rounds this grid down to the provided number of decimals.

        Parameters
        ----------
        decimals
            Decimals.

        Returns
        -------
        A new grid with rounded vectors.
        """
        return self.__class__(
            super().rounded(decimals),
            tuple(np.around(i, decimals=decimals) for i in self.coordinates),
            self.values,
            meta=self.meta,
        )

    def as_cell(self) -> cell.Cell:
        """
        Converts this grid into a unit cell.

        Returns
        -------
        A new cell including points from this grid.
        """
        v = self.values.reshape((-1,) + self.values.shape[len(self.coordinates):])
        return cell.Cell(self, self.explicit_coordinates().reshape(-1, len(self.vectors)), v, meta=self.meta)

    def interpolate_to_array(self, points: ndarray, driver=None, periodic: bool = True, **kwargs) -> ndarray:
        """
        Interpolates between point values in this grid.

        Parameters
        ----------
        points
            Target points in crystal basis.
        driver
            Interpolation driver.
        periodic
            If True, employs the periodicity of this cell while interpolating.
        kwargs
            Interpolation driver arguments.

        Returns
        -------
        An array with the interpolated data.
        """
        if driver is None:
            from scipy import interpolate
            driver = interpolate.interpn

        points = np.asanyarray(points)
        normalized = self.normalized()

        if periodic:

            data_points = list(normalized.coordinates)
            data_values = normalized.values

            # Avoid edge problems
            for i, a in enumerate(data_points):  # TODO: avoid changing data_points
                data_points[i] = np.insert(a, (0, a.size), (a[-1] - 1.0, a[0] + 1.0))

                left_slice = (slice(None),) * i + ((0,),) + (slice(None),) * (len(data_points) - i - 1)
                left = data_values[left_slice]

                right_slice = (slice(None),) * i + ((-1,),) + (slice(None),) * (len(data_points) - i - 1)
                right = data_values[right_slice]

                data_values = np.concatenate((right, data_values, left), axis=i)

            points = points % 1

        else:

            data_points = normalized.coordinates
            data_values = normalized.values

        # Interpolate
        return derived_from(driver(data_points, data_values, points, **kwargs), self.values)

    def interpolate_to_grid(self, points: list, **kwargs) -> Grid:
        """
        Interpolates between point values in this grid.
        Same as `interpolate_to_array` but takes grid points
        as an input and returns a grid object.

        Parameters
        ----------
        points
            Target grid points in crystal basis.
        kwargs
            Other arguments to `interpolate_to_array`.

        Returns
        -------
        A grid with the interpolated data.
        """
        return self.__class__(self, points, self.interpolate_to_array(grid_coordinates(points), **kwargs))

    def interpolate_to_cell(self, points: ndarray, **kwargs) -> cell.Cell:
        """
        Interpolates between point values in this grid.
        Same as `interpolate_to_array` but return a cell object.

        Parameters
        ----------
        points
            Target grid points in crystal basis.
        kwargs
            Other arguments to `interpolate_to_array`.

        Returns
        -------
        A cell with the interpolated data.
        """
        return cell.Cell(self, points, self.interpolate_to_array(points, **kwargs))

    def interpolate_to_path(self, nodes: Union[list, tuple, ndarray], n: int,
                            skip_segments: Union[list, tuple, ndarray] = None, **kwargs) -> cell.Cell:
        """
        Interpolates between point values in this grid.
        Same as `interpolate_to_array` but accepts nodes and
        point count to return interpolated points along the
        path as a cell object.

        Parameters
        ----------
        nodes
            A list or a 2D array of nodes' coordinates.
        n
            The desired point count in the path.
        skip_segments
            An optional array with segment indices to skip.
        kwargs
            Other arguments passed to `self.interpolate_to_cell`.

        Returns
        -------
        A cell object with interpolated data.
        """
        return self.interpolate_to_cell(generate_path(nodes, n, skip_segments=skip_segments), **kwargs)

    def tetrahedron_density(self, points: ndarray, resolved: bool = False, weights: ndarray = None) -> Union[ndarray, Grid]:
        """
        Computes the density of points' values (states).
        Uses the tetrahedron method from PRB 49, 16223 by E. Blochl et al.
        3D only.

        Parameters
        ----------
        points
            Values to calculate density at.
        resolved
            If True, returns a higher-dimensional tensor with spatially-
            and index-resolved density. The dimensions of the returned
            array are `self.values.shape + points.shape`.
        weights
            Assigns weights to points before computing the density.
            Only for `resolved=False`.

        Returns
        -------
        For `resolved=False` returns a 1D array with the density.
        For `resolved=True` returns the corresponding grid with
        values being the spatially-resolved density.
        """
        if len(self.vectors) != 3:
            raise ValueError("The tetrahedron density method is implemented only for 3D grids")
        if resolved and weights is not None:
            raise NotImplemented()

        points = np.asanyarray(points, dtype=np.float64)
        values = np.reshape(self.values, self.values.shape[:3] + (-1,))
        val_min = np.min(values, axis=(0, 1, 2))
        val_max = np.max(values, axis=(0, 1, 2))
        bottom = (val_max < points.min()).sum()
        top = (val_min < points.max()).sum()

        # Optimize size
        values = values[..., bottom:top]

        if resolved:
            return self.__class__(self, self.coordinates, compute_density_resolved(
                self.vectors, self.cartesian(), self.values, self.volume, points).reshape(values.shape + points.shape))

        else:
            if weights is None:
                weights = np.ones(self.values.shape, dtype=np.float64)

            if values.shape != weights.shape:
                raise ValueError(f"values.shape = {values.shape} != weights.shape = {weights.shape}")

            else:
                weights = np.asanyarray(weights, dtype=np.float64)
                weights = np.reshape(weights, weights.shape[:3] + (-1,))[..., bottom:top]
            return compute_density(self.vectors, self.cartesian(), self.values, self.volume, points, weights)
