from __future__ import annotations

from .basis import Basis
from .util import roarray, input_as_list, compute_angles, qhull_interpolation_driver, derived_from
from .attrs import check_vectors_inv, convert_vectors_inv, convert_coordinates, check_coordinates, convert_values,\
    check_values

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from attr import attrs, attrib

from typing import Union
from functools import cached_property


@attrs(frozen=True, eq=False)
class Cell(Basis):
    """Describes a unit cell."""
    coordinates = attrib(type=Union[ndarray, list, tuple], converter=convert_coordinates, validator=check_coordinates)
    values = attrib(type=Union[ndarray, list, tuple, str], converter=convert_values, validator=check_values)
    meta = attrib(type=dict, factory=dict, converter=dict)
    _vectors_inv = attrib(type=Union[ndarray, list, tuple], default=None, converter=convert_vectors_inv,
                          validator=check_vectors_inv)

    @classmethod
    def from_cartesian(cls, vectors: Union[ndarray, Basis, list, tuple], cartesian: Union[ndarray, list, tuple],
                       values: Union[ndarray, list, tuple, str], *args, proto: type = None,
                       vectors_inv: ndarray = None, **kwargs) -> Cell:
        """
        Constructs a cell using cartesian coordinates.

        Parameters
        ----------
        vectors : ndarray
            Cell basis.
        cartesian : ndarray
            A 2D array with cartesian coordinates.
        values : ndarray
            An array with values per each coordinate.
        args
            Other arguments.
        proto : class
            Class of the returned object.
        vectors_inv : ndarray
            Basis inverse.
        kwargs
            Other keyword arguments.

        Returns
        -------
        The resulting Cell.
        """
        basis = Basis(vectors, vectors_inv=vectors_inv)
        if vectors_inv is None:
            vectors_inv = basis.vectors_inv
        if proto is None:
            proto = cls
        return proto(basis, basis.transform_from_cartesian(cartesian), values, *args, vectors_inv=vectors_inv, **kwargs)

    @classmethod
    def random(cls, density: float, atoms: dict, shape: str = "box") -> Cell:
        """
        Prepares a cell with random coordinates.

        Parameters
        ----------
        density : float
            Atomic density.
        atoms : dict
            A dictionary with specimen-count pairs.
        shape : {"box"}
            The shape of the resulting cell.

        Returns
        -------
        result : Cell
            The resulting unit cell.
        """
        n_atoms = sum(atoms.values())
        coords = np.random.rand(n_atoms, 3)
        values = sum(([k] * v for k, v in atoms.items()), [])
        if shape == "box":
            a = (n_atoms / density) ** (1./3)
            return cls(np.eye(3) * a, coords, values)
        else:
            raise ValueError(f"Unknown shape={shape}")

    @cached_property
    def cartesian(self) -> ndarray:
        return roarray(self.transform_to_cartesian(self.coordinates))

    @cached_property
    def size(self) -> int:
        return len(self.coordinates)

    @cached_property
    def values_uq(self) -> ndarray:
        values_uq, values_encoded = np.unique(self.values, return_inverse=True, axis=0)
        self.__dict__["values_encoded"] = roarray(values_encoded.astype(np.int32))
        return roarray(values_uq)

    @cached_property
    def values_encoded(self) -> ndarray:
        _ = self.values_uq  # trigger attribute which sets this as well
        return self.values_encoded  # not a recursion

    @cached_property
    def values_lookup(self) -> dict:
        return dict(zip(self.values_uq, np.arange(len(self.values_uq))))

    def __eq__(self, other):
        return super().__eq__(other) and np.array_equal(self.coordinates, other.coordinates) and \
               np.array_equal(self.values, other.values)

    def normalized(self, left: float = 0, sort: Union[ndarray, str, int] = None) -> Cell:
        """
        Puts all points inside box boundaries and returns a copy.

        Parameters
        ----------
        left : float
            The left edge of the normalized box in cell
            coordinates. For example, ``left=-0.3`` stands
            for coordinates being placed in a ``[-0.3, 0.7)``
            interval.
        sort : ndarray
            An optional vector to sort along. Also accepts integers
            corresponding indicating basis vectors or one of 'xyz'
            to sort along cartesian axes.

        Returns
        -------
        result : Cell
            A copy of self with normalized coordinates.
        """
        d = self.state_dict(mark_type=False)
        d["coordinates"] = new_coordinates = ((self.coordinates - left) % 1) + left
        if sort is not None:
            if isinstance(sort, int):
                sort = self.vectors[sort]
            elif sort in ('x', 'y', 'z'):
                _sort = np.zeros(3)
                _sort['xyz'.index(sort)] = 1
                sort = _sort
            else:
                sort = np.asanyarray(sort)
            order = np.argsort(self.transform_to_cartesian(new_coordinates) @ sort)

            d["coordinates"] = d["coordinates"][order]
            d["values"] = d["values"][order]

        return self.__class__(**d)

    @input_as_list
    def distances(self, ids: list, cutoff: float = None, other: Union[Cell, ndarray] = None) -> Union[ndarray, csr_matrix]:
        """
        Computes distances between Cell points.

        Parameters
        ----------
        ids : ndarray
            Specimen IDs to compute distances between.
            Several shapes are accepted:
                * *empty*: returns a 2D matrix of all possible distances
                * nx2 array of ints: returns n distances between each pair
                  of [i, 0]-[i, 1] species;
                * 1D array of ints of length n: returns n-1 distances
                  between each pair of [i-1]-[i] species;
        cutoff : float
            Cutoff for obtaining distances. Only if ids is empty.
        other : Cell
            Other cell to compute distances to.

        Returns
        -------
        The resulting distance matrix in dense or sparse forms.
        """
        this = self.cartesian
        if other is None:
            other = this
        elif isinstance(other, Cell):
            other = other.cartesian

        if len(ids) == 0:
            if cutoff is None:
                return np.linalg.norm(this[:, None] - other[None, :], axis=-1)
            else:
                return KDTree(this).sparse_distance_matrix(KDTree(other), max_distance=cutoff)

        ids = np.asanyarray(ids, dtype=int)

        if ids.ndim == 1:
            if ids.shape[0] < 2:
                raise ValueError(f"Only {len(ids)} points are found, at least 2 required")
            return np.linalg.norm(this[ids[:-1], :] - other[ids[1:], :], axis=1)

        elif ids.ndim == 2:
            if ids.shape[1] != 2:
                raise ValueError(f"ids.shape={ids.shape}, required (n, 2)")
            return np.linalg.norm(this[ids[:, 0], :] - other[ids[:, 1], :], axis=1)

        else:
            raise ValueError(f"ids.ndim={ids.ndim}, required 1 or 2")

    def cartesian_delta(self, other: Cell, pbc: bool = True) -> ndarray:
        """
        Computes the distance between the corresponding pairs in two cells.

        Parameters
        ----------
        other : Cell
            Other cell to compute distance to.
        pbc : bool
            Periodic boundary conditions.

        Returns
        -------
        The resulting distances, one per pair.
        """
        assert self.size == other.size
        n_dims = len(self.vectors)
        if pbc:
            this_cartesian = self.normalized(-.5).cartesian
            other_replica = other.repeated(*(3,) * n_dims)
            other_replica_cartesian = other_replica.normalized(-.5).cartesian
            return np.min(np.linalg.norm(
                this_cartesian[None, ...] - other_replica_cartesian.reshape((3 ** n_dims,) + other.cartesian.shape),
                axis=-1,
            ), axis=0)
        else:
            return np.linalg.norm(self.cartesian - other.cartesian, axis=-1)

    def cartesian_copy(self, **kwargs) -> Cell:
        """
        Same as ``copy`` but accepts cartesian coordinates instead of crystal
        coordinates. Does exact same thing as ``copy`` if no arguments
        provided.

        Parameters
        ----------
        kwargs
            Arguments to ``self.from_cartesian``.

        Returns
        -------
        The resulting Cell.
        """
        state_dict = self.state_dict(mark_type=False)
        del state_dict["coordinates"]
        state_dict["cartesian"] = self.cartesian
        return self.from_cartesian(**{**state_dict, **kwargs})

    @input_as_list
    def angles(self, ids: ndarray) -> ndarray:
        """
        Computes angles between points in this cell.

        Parameters
        ----------
        ids : ndarray
            Point indexes to compute angles between.
            Several shapes are accepted:
                * nx3 array: computes n cosines of angles [i, 0]-[i, 1]-[i, 2];
                * 1D array of length n: computes n-2 cosines of angles along
                  the path ...-[i-1]-[i]-[i+1]-...;

        Returns
        -------
        An array with cosines.

        Examples
        --------
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
            >>> cell.angles((0, 1, 2)) # angle between vectors connecting {second and first} and {second and third} pts
            >>> cell.angles(0, 1, 2) # a simplified version of the above
            >>> cell.angles(0, 1, 3, 2) # two angles along path: 0-1-3 and 1-3-2
            >>> cell.angles(tuple(0, 1, 3, 2)) # same as the above
            >>> cell.angles((0, 1, 3),(1, 3, 2)) # same as the above
        """

        v = self.cartesian
        ids = np.asanyarray(ids, dtype=int)

        if len(ids.shape) == 1:
            if ids.shape[0] < 3:
                raise ValueError(f"Only {len(ids)} points are found, at least 3 required")
            vectors = v[ids[:-1], :] - v[ids[1:], :]
            nonzero = np.argwhere((vectors ** 2).sum(axis=1) > 0)[:, 0]
            if nonzero.shape[0] == 0:
                raise ValueError("All points coincide")

            vectors[:nonzero[0]] = vectors[nonzero[0]]
            vectors[nonzero[-1] + 1:] = vectors[nonzero[-1]]

            vectors_1 = vectors[:-1]
            vectors_2 = -vectors[1:]

            for i in range(nonzero.shape[0] - 1):
                vectors_1[nonzero[i] + 1:nonzero[i + 1]] = vectors_1[nonzero[i]]
                vectors_2[nonzero[i]:nonzero[i + 1] - 1] = vectors_2[nonzero[i + 1] - 1]

        elif len(ids.shape) == 2:
            if ids.shape[1] != 3:
                raise ValueError(f"ids.shape={ids.shape}, required (n, 3)")
            vectors_1 = v[ids[:, 0], :] - v[ids[:, 1], :]
            vectors_2 = v[ids[:, 2], :] - v[ids[:, 1], :]
        else:
            raise ValueError(f"ids.ndim={ids.ndim}, required 1 or 2")

        return compute_angles(vectors_1, vectors_2)

    def centered(self) -> Cell:
        """
        Generates a new cell where all points are shifted to maximize margins.

        Returns
        -------
        A new cell with centered coordinates.
        """
        sorted_coordinates = np.sort(self.coordinates % 1, axis=0)
        gaps = sorted_coordinates - np.roll(sorted_coordinates, 1, axis=0)
        gaps[0] = 1 - gaps[0]
        max_gap = np.argmax(gaps, axis=0)
        r = np.arange(self.coordinates.shape[1])
        gap_center = (self.coordinates[max_gap, r] + self.coordinates[max_gap - 1, r] + (max_gap == 0)) / 2
        return self.copy(coordinates=(self.coordinates - gap_center[None, :]) % 1)

    def ws_packed(self) -> Cell:
        """
        Generates a new cell where all points are replaced by their periodic images
        closest to the origin (i.e. appear inside Wigner-Seitz cell).

        Returns
        -------
        A new cell with packed coordinates.
        """
        result = self.normalized()
        cartesian = result.cartesian
        vertices = result.vertices

        d = cartesian[:, None, :] - vertices[None, :, :]
        d = (d ** 2).sum(axis=-1)
        d = np.argmin(d, axis=-1)

        return self.cartesian_copy(cartesian=cartesian - vertices[d, :])

    @input_as_list
    def isolated(self, gaps: list, units="crystal") -> Cell:
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
        gaps = np.array(gaps, dtype=float)
        if units == "cartesian":
            gaps /= ((self.vectors ** 2).sum(axis=1) ** .5)
        elif units == "crystal":
            pass
        else:
            raise ValueError(f"unknown units={units}")

        gaps += 1
        vectors = self.vectors * gaps[..., None]
        coordinates = self.coordinates / gaps[None, ...]
        coordinates += (0.5 * (gaps - 1) / gaps)[None, ...]
        return self.copy(vectors=vectors, coordinates=coordinates)

    def isolated2(self, gap: float) -> Cell:
        """
        Isolates points from their images in this cell by constructing a new
        larger orthorhombic cell.

        Parameters
        ----------
        gap : float
            The minimal gap size between the cloud of points and its periodic images.

        Returns
        -------
        An orthorhombic unit cell with the points.
        """
        c = self.normalized()
        cartesian = c.cartesian + gap
        shape = np.amax(c.vertices, axis=0) + 2 * gap
        return self.cartesian_copy(vectors=Basis.orthorhombic(shape), cartesian=cartesian)

    @input_as_list
    def select(self, piece: list) -> ndarray:
        """
        Selects points in this cell or grid inside a box in crystal basis.
        Images are not included.

        Parameters
        ----------
        piece : list
            Box dimensions ``[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]``,
            where x, y, z are basis vectors.

        Returns
        -------
        A numpy array with the selection mask.

        Examples
        --------
            >>> cell = Cell(Basis((1, 2, 3), kind="orthorhombic"), np.random.rand((4, 3)), np.arange(4))
            >>> cell.select((0,0,0,1,1,1)) # select all species with coordinates within (0,1) range
            >>> cell.select(0,0,0,1,1,1) # a simplified version of above
            >>> cell.select(0,0,0,0.5,1,1) # select the 'left' part
            >>> cell.select(0.5,0,0,1,1,1) # select the 'right' part
        """
        if not len(piece) == 2 * self.vectors.shape[0]:
            raise ValueError(f"len(piece) = {len(piece)} expected {2 * len(self.vectors)}")

        piece = np.reshape(piece, (2, -1))
        p1 = np.amin(piece, axis=0)
        p2 = np.amax(piece, axis=0)
        return np.all(self.coordinates < p2[None, :], axis=1) & np.all(self.coordinates >= p1[None, :], axis=1)

    @input_as_list
    def apply(self, selection) -> Cell:
        """
        Applies a mask to this cell or grid to keep a subset of points.

        Parameters
        ----------
        selection : list
            A bool mask with selected species.

        Returns
        -------
        The resulting cell.

        Examples
        --------
            >>> cell = Cell(Basis((1, 2, 3), kind="orthorhombic"), np.random.rand((4, 3)), np.arange(4))
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> result = cell.apply(selection) # Applies selection. Species outside the 'left' part are discarded.
        """
        selection = np.asanyarray(selection)
        return self.copy(coordinates=self.coordinates[selection, :], values=self.values[selection, ...])

    @input_as_list
    def discard(self, selection: list) -> Cell:
        """
        Discards points from this cell or grid according to the mask specified.
        Inverse of ``self.apply``.

        Parameters
        ----------
        selection : list
            Species to discard.

        Returns
        -------
        The resulting cell.

        Examples
        --------
            >>> cell = Cell(Basis.orthorhombic((1, 2, 3)), np.random.rand((4, 3)), np.arange(4))
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> result = cell.discard(selection) # Discards selection. Species inside the 'left' part are removed.
        """
        return self.apply(~np.asanyarray(selection))

    @input_as_list
    def cut(self, piece: list, select: Union[ndarray, list, tuple] = None) -> Cell:
        """
        Selects a box inside this cell or grid and returns it as a smaller cell.
        Basis vectors of the resulting instance are collinear to those of `self`.

        Parameters
        ----------
        piece : list
            Box dimensions ``[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]``,
            where x, y, z are basis vectors.
        select : list
            A custom selection mask or None if all points in the selected box
            have to be included.

        Returns
        -------
        A smaller instance with a subset of points.
        """
        if select is None:
            select = self.select(piece)

        piece = np.reshape(piece, (2, -1))
        p1 = np.amin(piece, axis=0)
        p2 = np.amax(piece, axis=0)

        vectors = self.vectors * (p2 - p1)[:, None]
        return self.cartesian_copy(vectors=vectors, cartesian=self.cartesian - p1 @ self.vectors).apply(select)

    @input_as_list
    def merge(self, cells: list) -> Cell:
        """
        Merges points from several unit cells with the same basis.

        Parameters
        ----------
        cells : list
            Cells to be merged.

        Returns
        -------
        A new unit cell with all points merged.
        """
        c = [self.coordinates]
        v = [self.values]

        for cell in cells:
            if not np.all(cell.vectors == self.vectors):
                raise ValueError(f'basis mismatch: {self.vectors} != {cell.vectors}')
            c.append(cell.coordinates)
            v.append(cell.values)

        return self.copy(coordinates=np.concatenate(c, axis=0), values=np.concatenate(v, axis=0))

    def stack(self, *cells: list, vector: int, **kwargs):
        """
        Stack multiple cells along the provided vector.

        Parameters
        ----------
        cells : list
            Cells and bases to stack.
        vector : int
            Basis vector to stack along.
        kwargs
            Other arguments to ``Basis.stack``.

        Returns
        -------
        The resulting cells stacked.
        """
        cells = (self, *cells)
        d = vector
        not_d = list(range(self.vectors.shape[0]))
        del not_d[d]
        dims = self.vectors.shape[0]

        basis = Basis.stack(*cells, vector=vector, **kwargs)

        values = np.concatenate(tuple(cell.values for cell in cells if isinstance(cell, Cell)), axis=0)

        cartesian = []
        shift = np.zeros(dims, dtype=float)
        for c in cells:
            if isinstance(c, Cell):
                # Fix for not-exactly-the-same vectors
                hvecs = c.vectors.copy()
                hvecs[not_d] = self.vectors[not_d]
                cartesian.append(Basis(hvecs).transform_to_cartesian(c.coordinates) + shift[None, :])
            shift += c.vectors[d, :]
        cartesian = np.concatenate(cartesian, axis=0)

        return self.__class__.from_cartesian(basis, cartesian, values)

    @input_as_list
    def supercell(self, vec: list) -> Cell:
        """
        Produces a supercell from this cell.

        Parameters
        ----------
        vec : ndarray
            New vectors expressed in this basis.

        Returns
        -------
        A supercell.

        Examples
        --------
            >>> cell = Cell(Basis.orthorhombic((1, 2, 3)), np.random.rand((4, 3)), np.arange(4))
            >>> s_cell = cell.supercell(np.eye(cell.size)) # returns a copy
            >>> r_cell = cell.supercell(np.diag((1, 2, 3))) # same as cell.repeated(1, 2, 3)
        """
        vec = np.array(vec, dtype=int)  # integer-valued supercell vectors
        vec_inv = np.linalg.inv(vec)
        vec_det = int(abs(np.linalg.det(vec)))
        cofactor = (vec_inv * vec_det).astype(int)  # integer-valued vector cofactor matrix
        gcd = np.array(list(
            np.gcd.reduce(i)
            for i in cofactor
        ))  # greatest common divisor of cofactor vectors
        axes_steps = vec_det // gcd  # diag(axes_steps) are (minimal) supercell "diagonal" vectors

        # compose the volume out of divisors of axes_steps
        volume = vec_det
        recipe = np.ones_like(axes_steps)
        for i, step in enumerate(axes_steps):
            g = np.gcd(volume, step)
            volume //= g
            recipe[i] = g
            if volume == 1:
                break
        else:
            raise RuntimeError("Failed to compose the equivalent diagonal supercell")

        return self.repeated(recipe).cartesian_copy(vectors=vec @ self.vectors).normalized()

    def species(self) -> dict:
        """
        Counts atomic species.

        Returns
        -------
        A dictionary with unique point values as keys and numbers of their occurrences as values.
        """
        answer = {}
        for s in self.values:
            try:
                answer[s] += 1
            except KeyError:
                answer[s] = 1
        return answer

    @input_as_list
    def transpose_vectors(self, new: list) -> Cell:
        """
        Reorders basis vectors without changing cartesian coordinates.

        Parameters
        ----------
        new : list
            The new order as a list of integers.

        Returns
        -------
        A new unit cell with reordered vectors.
        """
        return Cell(super().transpose_vectors(new), self.coordinates[:, new], self.values, meta=self.meta)

    def rounded(self, decimals: int = 8) -> Cell:
        """
        Rounds this Cell down to the provided number of decimals.

        Parameters
        ----------
        decimals : int
            Decimals.

        Returns
        -------
        A new Basis with rounded vectors.
        """
        return Cell(super().rounded(decimals), np.around(self.coordinates, decimals=decimals), self.values,
                    meta=self.meta)

    @input_as_list
    def interpolate(self, points: list, driver=None, periodic: bool = True, **kwargs) -> Cell:
        """
        Interpolates values between points in this cell and returns the interpolated Cell.

        Parameters
        ----------
        points : list
            Interpolation points in this basis.
        driver : Callable
            Interpolation driver.
        periodic : bool
            If True, interpolates data in periodic boundary conditions.
        kwargs
            Driver arguments.

        Returns
        -------
        A new unit cell with the interpolated data.
        """
        points = np.asanyarray(points, dtype=float)

        if driver is None:
            driver = qhull_interpolation_driver

        if periodic:

            # Avoid edge problems by creating copies of this cell
            supercell = self.repeated((3,) * self.vectors.shape[0]).normalized()

            data_points = supercell.cartesian
            data_values = supercell.values

            # Shift points to the central unit cell
            points_i = self.transform_to_cartesian(points % 1) + self.vectors.sum(axis=0)[None, :]

        else:

            data_points = self.cartesian
            data_values = self.values
            points_i = self.transform_to_cartesian(points)

        # Interpolate
        return self.__class__(self, points, derived_from(driver(data_points, data_values, points_i, **kwargs), self.values))
