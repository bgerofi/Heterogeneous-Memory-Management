import numpy as np
import unittest
from gym.spaces import Space


def slen(s: slice):
    return s.stop - s.start


class WindowObservationSpace(Space):
    def __init__(self, nrows, ncols, seed=None):

        if nrows <= 0 or ncols <= 0:
            raise ValueError("Invalid input. nrows > 0 and ncols > 0.")

        shape = [nrows, ncols]
        super().__init__(shape, np.int64, seed)

    def __eq__(self, other):
        if not isinstance(other, WindowObservationSpace):
            return False
        if other.shape != self.shape:
            return False
        return True

    def empty(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def from_sparse_matrix(self, x, y):
        def scale(x, low=0, high=1):
            max_x = np.nanmax(y)
            min_x = np.nanmin(y)
            x = np.array(x).copy()

            if max_x == min_x:
                x.fill((high - low) / 2)
                return x
            if min_x != 0:
                x = x - min_x
            if (max_x - min_x) != 1:
                x = x / (max_x - min_x)
            if (high - low) != 1:
                x = x * (high - low)
            if low != 0:
                x = x + low
            return x

        # Scale axis and convert to integer values in shape bounds.
        y = scale(y, 0, self.shape[1]).astype(np.uint64)
        y = np.minimum(y, self.shape[1] - 1)
        x = scale(x, 0, self.shape[0]).astype(np.uint64)
        x = np.minimum(x, self.shape[0] - 1)

        # Accumulate set values in empty matrix cells.
        ndarray = self.empty()
        for i, j in zip(x, y):
            ndarray[i, j] += 1

        return ndarray

    def sample(self) -> np.ndarray:
        return self.np_random.sample(self.shape, dtype=self.dtype)

    def contains(self, x) -> bool:
        if isinstance(x, TraceSet):
            return True
        elif isinstance(x, np.ndarray):
            return len(x.shape) == 2
        else:
            return False
