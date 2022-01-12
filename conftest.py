import numpy as np
from pycoordinates import Basis, Cell, Grid
from itertools import zip_longest


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, Basis) and isinstance(right, Basis) and op == "==":
        output = [f"{type(left).__name__} (L) == {type(right).__name__} (R)"]
        if type(left) is not type(right):
            return output

        arrays = [("vectors", left.vectors, right.vectors)]
        if isinstance(left, Cell) and isinstance(right, Cell):
            arrays.append(("coordinates", left.coordinates, right.coordinates))
            arrays.append(("values", left.values, right.values))
        elif isinstance(left, Grid) and isinstance(right, Grid):
            for x, i, j in enumerate(zip_longest(left.coordinates, right.coordinates)):
                arrays.append((f"coordinates[{x}]", i, j))
            arrays.append(("values", left.values, right.values))

        def _is_none(a):
            if a is None:
                return "None"
            else:
                return "not None"

        for name, left, right in arrays:
            if left is None or right is None:
                output.append(f"  {name} L is {_is_none(left)} != R is {_is_none(right)}")
            eq = np.array_equal(left, right)
            if eq:
                output.append(f"  {name} are equal")
            else:
                output.extend([
                    f"  {name} are not equal",
                    f"    L {left}",
                    f"    R {right}",
                ])
        return output
