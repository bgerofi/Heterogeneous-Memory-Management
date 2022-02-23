import numpy as np
import unittest
from gym.spaces import Space


def slen(s: slice):
    return s.stop - s.start


class WindowObservationSpace(Space):
    def __init__(self, num_samples, num_addresses, seed=None):

        if num_samples <= 0 or num_addresses <= 0:
            raise ValueError("Invalid input. num_samples > 0 and num_addresses > 0.")

        shape = [num_samples, num_addresses]
        super().__init__(shape, np.float64, seed)

    def copy(self):
        obs = WindowObservationSpace.__new__()
        obs._shape = self.shape
        obs.dtype = self.dtype
        obs._np_random = self._np_random
        obs.seed = self.seed
        return obs

    def from_ndarray(self, ndarray):
        if len(ndarray.shape) != 2:
            raise ValueError(
                "Invalid ndarray for observation. Array must have 2 dimensions"
            )
        if ndarray.dtype != self.dtype:
            ndarray = ndarray.astype(self.dtype)

        # Scale rows
        if ndarray.shape[0] > self.shape[0]:
            _ndarray = np.empty((self.shape[0], ndarray.shape[1]), dtype=self.dtype)
            WindowObservationSpace._shrink_rows(ndarray, _ndarray)
        elif ndarray.shape[0] < self.shape[0]:
            _ndarray = np.empty((self.shape[0], ndarray.shape[1]), dtype=self.dtype)
            WindowObservationSpace._grow_rows(ndarray, _ndarray)
        else:
            _ndarray = ndarray

        # Scale columns
        if ndarray.shape[1] > self.shape[1]:
            ndarray = np.empty(self.shape, dtype=self.dtype)
            WindowObservationSpace._shrink_columns(_ndarray, ndarray)
        elif ndarray.shape[1] < self.shape[1]:
            ndarray = np.empty(self.shape, dtype=self.dtype)
            WindowObservationSpace._grow_columns(_ndarray, ndarray)
        else:
            ndarray = _ndarray

        return ndarray

    def from_addresses(self, vaddr):
        # Use offsets instead of addresses
        vaddr = vaddr - np.nanmin(vaddr)
        # Create the array of observations with no observed sample
        ndarray = np.zeros((len(vaddr), np.nanmax(vaddr)), dtype=self.dtype)
        # Set samples with a default value.
        for i, addr in enumerate(vaddr):
            ndarray[i, addr] = 1.0
        # Scale the ndarray to fit this observation shape.
        return self.from_ndarray(ndarray)

    def from_window(self, window):
        vaddr = np.array(window.trace_ddr.virtual_addresses(), dtype=np.int64)
        return self.from_addresses(vaddr)

    @staticmethod
    def _shrink_rows(from_array, to_array):
        from_shape = from_array.shape
        to_shape = to_array.shape

        p_i = from_shape[0] // to_shape[0]
        q_i = from_shape[0] - p_i * to_shape[0]

        for i in range(q_i):
            slice_i = slice(i * (p_i + 1), (i + 1) * (p_i + 1))
            to_array[i, :] = np.sum(from_array[slice_i, :], 0)

        offset = q_i * (p_i + 1)
        for i in range(q_i, to_shape[0] - q_i):
            slice_i = slice(offset + p_i * i, offset + p_i * (i + 1))
            to_array[i, :] = np.sum(from_array[slice_i, :], 0)

    @staticmethod
    def _shrink_columns(from_array, to_array):
        from_shape = from_array.shape
        to_shape = to_array.shape

        p_j = from_shape[1] // to_shape[1]
        q_j = from_shape[1] - p_j * to_shape[1]

        for j in range(q_j):
            slice_j = slice(j * (p_j + 1), (j + 1) * (p_j + 1))
            to_array[:, j] = np.sum(from_array[:, slice_j], 1)

        offset = q_j * (p_j + 1)
        for j in range(q_j, to_shape[1] - q_j):
            slice_j = slice(offset + p_j * j, offset + p_j * (j + 1))
            to_array[:, j] = np.sum(from_array[:, slice_j], 1)

    @staticmethod
    def _grow_rows(from_array, to_array):
        from_shape = from_array.shape
        to_shape = to_array.shape

        p_i = to_shape[0] // from_shape[0]
        q_i = to_shape[0] - p_i * from_shape[0]

        for i in range(q_i):
            slice_i = slice(i * (p_i + 1), (i + 1) * (p_i + 1))
            to_array[slice_i, :] = from_array[i, :] / slen(slice_i)

        offset = q_i * (p_i + 1)
        for i in range(q_i, from_shape[0] - q_i):
            slice_i = slice(offset + p_i * i, offset + p_i * (i + 1))
            to_array[slice_i, :] = from_array[i, :] / slen(slice_i)

    @staticmethod
    def _grow_columns(from_array, to_array):
        from_shape = from_array.shape
        to_shape = to_array.shape

        p_j = to_shape[1] // from_shape[1]
        q_j = to_shape[1] - p_j * from_shape[1]

        for j in range(q_j):
            slice_j = slice(j * (p_j + 1), (j + 1) * (p_j + 1))
            to_array[:, slice_j] = (from_array[:, j] / slen(slice_j)).reshape(
                (from_shape[0], 1)
            )

        offset = q_j * (p_j + 1)
        for j in range(q_j, from_shape[1] - q_j):
            slice_j = slice(offset + p_j * j, offset + p_j * (j + 1))
            to_array[:, slice_j] = (from_array[:, j] / slen(slice_j)).reshape(
                (from_shape[0], 1)
            )

    def sample(self) -> np.ndarray:
        return self.np_random.sample(self.shape, dtype=self.dtype)

    def contains(self, x) -> bool:
        if isinstance(x, TraceSet):
            return True
        elif isinstance(x, ndarray):
            return x.shape == self.shape
        else:
            return False


class TestWindowObservation(unittest.TestCase):
    def test_same_size(self):
        array = np.empty((2, 2))
        array[:] = 1
        obs_array = WindowObservationSpace(2, 2).from_ndarray(array)
        self.assertTrue(array.tolist() == obs_array.tolist())

    def test_grow_lines(self):
        array = np.empty((2, 2))
        array[:] = 1
        comp_array = np.empty((4, 2))
        comp_array[:] = 0.5
        obs_array = WindowObservationSpace(4, 2).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())

    def test_grow_columns(self):
        array = np.empty((2, 2))
        array[:] = 1
        comp_array = np.empty((2, 4))
        comp_array[:] = 0.5
        obs_array = WindowObservationSpace(2, 4).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())

    def test_grow(self):
        array = np.empty((2, 2))
        array[:] = 1
        comp_array = np.empty((4, 4))
        comp_array[:] = 0.25
        obs_array = WindowObservationSpace(4, 4).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())

    def test_shrink_lines(self):
        array = np.empty((4, 4))
        array[:] = 1
        comp_array = np.empty((2, 4))
        comp_array[:] = 2
        obs_array = WindowObservationSpace(2, 4).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())

    def test_shrink_columns(self):
        array = np.empty((4, 4))
        array[:] = 1
        comp_array = np.empty((4, 2))
        comp_array[:] = 2
        obs_array = WindowObservationSpace(4, 2).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())

    def test_shrink(self):
        array = np.empty((4, 4))
        array[:] = 1
        comp_array = np.empty((2, 2))
        comp_array[:] = 4
        obs_array = WindowObservationSpace(2, 2).from_ndarray(array)
        self.assertTrue(comp_array.tolist() == obs_array.tolist())


if __name__ == "__main__":
    unittest.main()
