from __future__ import annotations

from .basis import Basis
from .util import roarray
from .attrs import check_vectors_inv, convert_vectors_inv, convert_coordinates, check_coordinates, convert_values,\
    check_values

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from attr import attrs, attrib, asdict

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

    @cached_property
    def cartesian(self) -> ndarray:
        return roarray(self.transform_to_cartesian(self.coordinates))

    @cached_property
    def size(self) -> int:
        return len(self.coordinates)

    @cached_property
    def values_uq(self) -> ndarray:
        values_uq, values_encoded = np.unique(self.values, return_inverse=True)
        self.__dict__["values_encoded"] = roarray(values_encoded.astype(np.int32))
        return roarray(values_uq)

    @cached_property
    def values_encoded(self) -> ndarray:
        _ = self.values_uq  # trigger attribute which sets this as well
        return self.values_encoded  # not a recursion

    @cached_property
    def values_lookup(self) -> dict:
        return dict(zip(self.values_uq, np.arange(len(self.values_uq))))

    def normalized(self, left: float = 0) -> Cell:
        """
        Puts all points inside box boundaries and returns a copy.

        Parameters
        ----------
        left : float
            The left edge of the normalized box in cell
            coordinates. For example, ``left=-0.3`` stands
            for coordinates being placed in a ``[-0.3, 0.7)``
            interval.

        Returns
        -------
        result : Cell
            A copy of self with normalized coordinates.
        """
        d = {(k[1:] if k.startswith("_") else k): v for k, v in asdict(self).items()}
        d["coordinates"] = ((self.coordinates - left) % 1) + left
        return self.__class__(**d)

    def distances(self, cutoff: float = None, other: Union[Cell, ndarray] = None) -> Union[ndarray, csr_matrix]:
        """
        Computes distances between Cell points.

        Parameters
        ----------
        cutoff : float
            Cutoff for obtaining distances.
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

        if cutoff is None:
            return np.linalg.norm(this[:, np.newaxis] - other[np.newaxis, :], axis=-1)
        else:
            this = KDTree(this)
            other = KDTree(other)
            return this.sparse_distance_matrix(other, max_distance=cutoff)

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

    def repeated(self, *args) -> Cell:
        """
        Increases the Cell by cloning it along all vectors.

        Parameters
        ----------
        *args
            Repeat counts along each vector.

        Returns
        -------
        The resulting bigger Cell.
        """
        args = np.array(args, dtype=int)
        x = np.prod(args)
        vectors = self.vectors * args[np.newaxis, :]
        coordinates = np.tile(self.coordinates, (x, 1))
        coordinates /= args[np.newaxis, :]
        coordinates.shape = (x, *self.coordinates.shape)
        values = np.tile(self.values, x)
        shifts = list(np.linspace(0, 1, i, endpoint=False) for i in args)
        shifts = np.meshgrid(*shifts)
        shifts = np.stack(shifts, axis=-1)
        shifts = shifts.reshape(-1, shifts.shape[-1])
        coordinates += shifts[:, np.newaxis, :]
        coordinates = coordinates.reshape(-1, coordinates.shape[-1])
        return Cell(vectors, coordinates, values)

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
