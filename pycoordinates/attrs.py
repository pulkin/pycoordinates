from .util import ro_float_array_copy, roarray_copy

from numpy import ndarray

from typing import Union


def convert_vectors(obj) -> ndarray:
    if isinstance(obj, (ndarray, list, tuple)):
        return ro_float_array_copy(obj)
    elif "vectors" in dir(obj):
        return ro_float_array_copy(obj.vectors)
    else:
        raise ValueError(f"failed to convert to vectors: {obj}")


def check_vectors(instance, attribute: str, value: ndarray):
    if value.ndim != 2:
        raise ValueError("vectors have to be a 2D array")


def convert_vectors_inv(obj: Union[None, ndarray]) -> ndarray:
    if obj is not None:
        return ro_float_array_copy(obj)


def check_vectors_inv(instance, attribute: str, value: ndarray):
    if value is not None and instance.vectors.shape != value.T.shape:
        raise ValueError(f"vectors_inv.shape={value.shape} is different from "
                         f"vectors.T.shape={instance.vectors.T.shape}")


def convert_coordinates(coordinates: Union[ndarray, list, tuple]) -> ndarray:
    coordinates = ro_float_array_copy(coordinates)
    if coordinates.ndim == 1:
        coordinates.shape = (1,) + coordinates.shape
    return coordinates


def check_coordinates(instance, attribute: str, coordinates: ndarray):
    dims = len(instance.vectors)
    if coordinates.ndim != 2:
        raise ValueError(f"coordinates.shape={coordinates.shape} is not a 2D array")
    if coordinates.shape[1] != dims:
        raise ValueError(f'coordinates.shape={coordinates.shape}, expected ({dims},) or (*, {dims})')


def convert_values(values: Union[ndarray, list, tuple, str]) -> ndarray:
    values = roarray_copy(values)
    if values.ndim == 0:
        values.shape = (1, )
    return values


def check_values(instance, attribute: str, values: ndarray):
    if len(values) != len(instance.coordinates):
        raise ValueError(f'values.shape = {values.shape} does not match coordinates.shape={instance.coordinates.shape}')

