from __future__ import annotations

from .util import roarray, compute_angles
from .attrs import check_vectors, convert_vectors, check_vectors_inv, convert_vectors_inv

import numpy as np
from numpy import ndarray, array, diag
from attr import attrs, attrib, asdict

from typing import Union
from functools import cached_property
from itertools import product
from warnings import warn


@attrs
class Identifiable:
    """Objects that can be saved and loaded."""

    @classmethod
    def class_id(cls) -> str:
        """Retrieves a unique ID of the class."""
        return cls.__module__ + "." + getattr(cls, "__qualname__", cls.__name__)

    def state_dict(self, mark_type: bool = True) -> dict:
        """
        Prepares a state dict of this object.

        Parameters
        ----------
        mark_type : bool
            Include the 'type' field into the resulting dictionary.

        Returns
        -------
        A dictionary with the state.
        """
        result = asdict(self, filter=lambda attr, value: not attr.name.startswith("_"))
        if mark_type:
            result["type"] = self.class_id()
        return result

    @classmethod
    def from_state_dict(cls, data: dict, check_type: bool = "warn") -> Identifiable:
        """
        Restores the object from state dict data.

        Parameters
        ----------
        data: dict
            State data.
        check_type : bool
            If set, checks the 'type' field.

        Returns
        -------
        The resulting object.
        """
        data = dict(data)
        if check_type:
            t = data.get("type", None)
            if t != cls.class_id():
                msg = f"Invalid type {t}, expected type {cls.class_id()}"
                if check_type == "warn":
                    warn(msg)
                else:
                    raise TypeError(msg)
        if "type" in data:
            del data["type"]
        return cls(**data)

    @classmethod
    def from_state_data(cls, data: Union[list, dict]) -> Union[list, Identifiable]:
        """
        Restores the object or multiple objects from state dict data.

        Parameters
        ----------
        data: dict
            State data.

        Returns
        -------
        The resulting object or many objects.
        """
        if isinstance(data, dict):
            return cls.from_state_dict(data)
        elif isinstance(data, list):
            return list(map(cls.from_state_dict, data))
        else:
            raise ValueError(f"Unknown data type to restore from: {type(data)}")

    def copy(self, **kwargs) -> Identifiable:
        """
        Computes a copy.

        Parameters
        ----------
        kwargs
            Fields to update.

        Returns
        -------
        The resulting copy.
        """
        return self.from_state_dict({**self.state_dict(), **kwargs})


@attrs(frozen=True, cmp=False)
class Basis(Identifiable):
    vectors = attrib(type=Union[ndarray, "Basis", list, tuple], converter=convert_vectors, validator=check_vectors)
    meta = attrib(type=dict, factory=dict, converter=dict)
    _vectors_inv = attrib(type=Union[ndarray, list, tuple], default=None, converter=convert_vectors_inv,
                          validator=check_vectors_inv)
    """
    A class describing a set of vectors forming a vector basis.

    Parameters
    ----------
    vectors
        A matrix of basis vectors.
    meta
        Metadata.
    vectors_inv
        Optional pre-computed inverse vectors.
    """
    @classmethod
    def orthorhombic(cls, lengths: ndarray, **kwargs) -> Basis:
        """
        An orthorhombic basis.

        Parameters
        ----------
        lengths
            A 1D array with vectors lengths.
        kwargs
            Other init arguments.

        Returns
        -------
        The resulting orthorhombic basis.
        """
        return cls(diag(lengths), **kwargs)

    @classmethod
    def triclinic(cls, lengths: ndarray, cosines: ndarray, **kwargs) -> Basis:
        """
        A triclinic basis defined through vector lengths and angles.

        Parameters
        ----------
        lengths
            A 1D array with 3 vectors lengths.
        cosines
            A 1D array with 3 vector angle cosines.
        kwargs
            Other init arguments.

        Returns
        -------
        The resulting basis.
        """
        lengths = array(lengths)
        cosines = array(cosines)
        assert lengths.shape == (3,), "Only 3-vectors are accepted as lengths"
        assert cosines.shape == (3,), "Only 3-vectors are accepted as cosines"
        volume = lengths[0] * lengths[1] * lengths[2] * (
                1 + 2 * cosines[0] * cosines[1] * cosines[2] - cosines[0] ** 2 - cosines[1] ** 2 - cosines[2] ** 2
        ) ** .5
        sines = (1 - cosines ** 2) ** .5
        height = volume / lengths[0] / lengths[1] / sines[2]
        vectors = array((
            (lengths[0], 0, 0),
            (lengths[1] * cosines[2], lengths[1] * sines[2], 0),
            (lengths[2] * cosines[1], abs((lengths[2] * sines[1]) ** 2 - height ** 2) ** .5, height)
        ))
        return cls(vectors, **kwargs)

    @classmethod
    def diamond(cls, a: float, **kwargs) -> Basis:
        """
        Diamond basis.

        Parameters
        ----------
        a
            Lattice constant.
        kwargs
            Other init arguments.

        Returns
        -------
        The resulting diamond basis.
        """
        a = 0.5 * a
        return Basis([[0, a, a], [a, 0, a], [a, a, 0]], **kwargs)

    @cached_property
    def vectors_inv(self) -> ndarray:
        if self._vectors_inv is not None:
            return self._vectors_inv
        return roarray(np.linalg.inv(self.vectors))

    @cached_property
    def det(self):
        return float(np.linalg.det(self.vectors))

    @cached_property
    def volume(self) -> float:
        return abs(self.det)

    @cached_property
    def vertices(self) -> ndarray:
        result = []
        for v in product((0.0, 1.0), repeat=self.vectors.shape[0]):
            result.append(self.transform_to_cartesian(np.asanyarray(v)))
        return roarray(np.asanyarray(result))

    @cached_property
    def edges(self) -> ndarray:
        result = []
        for e in range(self.vectors.shape[0]):
            for v in product((0.0, 1.0), repeat=self.vectors.shape[0] - 1):
                v1 = v[:e] + (0.,) + v[e:]
                v2 = v[:e] + (1.,) + v[e:]
                result.append((
                    (self.vectors * np.asanyarray(v1)[:, None]).sum(axis=0),
                    (self.vectors * np.asanyarray(v2)[:, None]).sum(axis=0),
                ))
        return roarray(np.asanyarray(result))

    @cached_property
    def reciprocal(self) -> Basis:
        return Basis(self.vectors_inv.T)

    def __eq__(self, other):
        return type(self) == type(other) and np.array_equal(self.vectors, other.vectors)

    def transform_matrix(self, to: Basis) -> ndarray:
        """
        Prepares a transform matrix.

        Parameters
        ----------
        to
            Basis to transform to.

        Returns
        -------
        The resulting transformation matrix.
        """
        return self.vectors @ to.vectors_inv

    def transform_to(self, to: Basis, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates to another basis set.

        Parameters
        ----------
        to
            The new basis to transform to.
        coordinates
            Array of coordinates to be transformed.

        Returns
        -------
        An array with transformed coordinates.
        """
        return np.tensordot(
            coordinates,
            self.transform_matrix(to),
            axes=((len(coordinates.shape) - 1,), (0,))
        )

    def transform_from(self, fr: Basis, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates from another basis set.

        Parameters
        ----------
        fr
            Basis to transform from.
        coordinates
            Array of coordinates to be transformed.

        Returns
        -------
        An array with transformed coordinates.
        """
        return fr.transform_to(self, coordinates)

    def transform_to_cartesian(self, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates to cartesian.

        Parameters
        ----------
        coordinates
            Array of coordinates to be transformed.

        Returns
        -------
        An array with transformed coordinates.
        """
        return self.transform_to(Basis(np.eye(self.vectors.shape[0])), coordinates)

    def transform_from_cartesian(self, coordinates: ndarray) -> ndarray:
        """
        Transforms coordinates from cartesian.

        Parameters
        ----------
        coordinates
            Array of coordinates to be transformed.

        Returns
        -------
        An array with transformed coordinates.
        """
        return self.transform_from(Basis(np.eye(self.vectors.shape[0])), coordinates)

    def rotated(self, axis: ndarray, angle: float, units: str = 'rad') -> Basis:
        """
        Rotates this basis.

        Parameters
        ----------
        axis
            Axis to rotate around.
        angle
            Angle to rotate.
        units
            Angle units: radians, degree or fractional.

        Returns
        -------
        A rotated copy of this basis.
        """
        units = {
            "rad": 1.0,
            "deg": np.pi / 180,
            "frac": np.pi * 2,
        }[units]
        angle *= units
        c = np.cos(angle)
        s = np.sin(angle)
        axis = axis / (axis ** 2).sum() ** .5
        axis_x = np.asanyarray((
            (0, -axis[2], axis[1]),
            (axis[2], 0, -axis[0]),
            (-axis[1], axis[0], 0),
        ))
        rot_matrix = c * np.eye(self.vectors.shape[0]) + s * axis_x + (1 - c) * np.dot(axis[:, None], axis[None, :])
        return Basis(self.vectors @ rot_matrix)

    def stack(self, *other, vector: Union[str, int] = 'x', tolerance: float = 1e-10,
              restrict_collinear: bool = False) -> Basis:
        """
        Stacks several bases along one of the vectors.

        Parameters
        ----------
        other
            Other bases to stack.
        vector
            Vector along which to stack, either integer or one of 'xyz'
            corresponding to vectors 0, 1, 2.
        tolerance
            The error threshold when determining equality of non-stacking
            vectors.
        restrict_collinear
            If True will perform a check whether stacking vectors are
            collinear and raise ValueError if they are not.

        Returns
        -------
        Stacked basis.
        """
        other = self, *other
        d = {'x': 0, 'y': 1, 'z': 2}.get(vector, vector)

        other_vectors = list(range(other[0].vectors.shape[0]))
        del other_vectors[d]

        # 3d array with lattice vectors: shapes[i,j,k] i=cell, j=lattice vector, k = component
        shapes = np.concatenate(tuple(i.vectors[None, ...] for i in other), axis=0)

        # Check if non-stacking lattice vectors coincide
        stacking_vectors_sum = shapes[:, d, :].sum(axis=0)
        vec_lengths = (shapes ** 2).sum(axis=2) ** 0.5
        other_vectors_d = shapes[:, other_vectors, :] - shapes[0, other_vectors, :][None, ...]
        other_vectors_ds = (other_vectors_d ** 2).sum(axis=-1) ** .5

        if np.any(other_vectors_ds > tolerance * vec_lengths[:, other_vectors]):
            raise ValueError(
                'Dimension mismatch for stacking:\n{}\nCheck your input basis vectors or set tolerance to at least {} '
                'to silence this exception'.format(
                    shapes,
                    np.amax(other_vectors_ds / vec_lengths[:, other_vectors]),
                ))

        if restrict_collinear:
            angles = np.abs(compute_angles(shapes[:, d, :], stacking_vectors_sum[None, :]) - 1)
            if np.any(angles > tolerance):
                raise ValueError('Vectors to stack along are not collinear:\n{}\nCheck your input basis vectors or set '
                                 'tolerance to at least {} to silence this exception'.format(shapes, np.amax(angles)))

        shape = self.vectors.copy()
        shape[d, :] = stacking_vectors_sum
        return Basis(shape, meta=self.meta)

    def repeated(self, *times) -> Basis:
        """
        Produces a new instance from a given one by repeating it along
        all vectors the given numbers of times.

        Parameters
        ----------
        times
            Integers specifying repetition counts.

        Returns
        -------
        Basis repeated along its vectors.
        """
        result = self
        for i, t in enumerate(times):
            result = result.stack(*((result,) * (t - 1)), vector=i)
        return result

    def rounded(self, decimals: int = 8) -> Basis:
        """
        Rounds this Basis down to the given number of decimals.

        Parameters
        ----------
        decimals
            Decimals.

        Returns
        -------
        A new Basis with rounded vectors.
        """
        return Basis(np.around(self.vectors, decimals=decimals), meta=self.meta)
