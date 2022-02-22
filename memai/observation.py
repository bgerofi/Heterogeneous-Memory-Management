import numpy as np
from memai import TraceSet
from gym.spaces import Space


def slen(s: slice):
    return s.stop - s.start


class WindowObservation(Space[np.ndarray]):
    def __init__(self, num_addresses, num_samples, seed=None):

        if num_samples <= 0 or num_addresses <= 0:
            raise ValueError("Invalid input. num_samples > 0 and num_addresses > 0.")

        shape = [num_addresses, num_samples]
        super().__init__(shape, np.float64, seed)
        self._ndarray = np.empty(shape, dtype=self.dtype)

    def copy(self):
        obs = WindowObservation.__new__()
        obs._shape = self._shape
        obs.dtype = self.dtype
        obs._np_random = self._np_random
        obs.seed = self.seed
        obs._ndarray = self._ndarray.copy()
        return obs

    @property
    def ndarray(self):
        return self._ndarray

    def num_addresses(self):
        return self.shape()[0]

    def num_samples(self):
        return self.shape()[1]

    def zero(self):
        self._ndarray = np.zeros(self.shape(), dtype=self.dtype)
        return self

    def from_ndarray(self, ndarray):
        if len(ndarray.shape) != 2:
            raise ValueError(
                "Invalid ndarray for observation. Array must have 2 dimensions"
            )

        num_addresses = ndarray.shape[0]
        num_samples = ndarray.shape[1]

        # Scale addresses
        if num_addresses > self.num_addresses():
            addr_ndarray = np.empty((self.num_addresses(), num_samples))
            WindowObservation._shrink_addresses(ndarray, addr_ndarray)
        elif num_addresses < self.num_addresses():
            addr_ndarray = np.empty((self.num_addresses(), num_samples))
            WindowObservation._grow_addresses(ndarray, addr_ndarray)
        else:
            addr_ndarray = ndarray

        # Scale samples
        if num_samples > self.num_samples():
            WindowObservation._shrink_samples(addr_ndarray, self._ndarray)
        elif num_samples < self.num_samples():
            WindowObservation._grow_samples(addr_ndarray, self._ndarray)
        else:
            self._ndarray = addr_ndarray
        return self

    def from_window(self, window):
        if not isinstance(window, TraceSet):
            raise ValueError(
                "Invalid window type {}. Expected TraceSet.".format(type(window))
            )

        vaddr = np.array(window.trace_ddr.virtual_addresses(), dtype=np.int64)
        # Use offsets instead of addresses
        vaddr = vaddr - min(vaddr)
        # Create the array of observations with no observed sample
        ndarray = np.zeros((max(vaddr), len(vaddr)), dtype=self.dtype)
        # Set samples with a default value.
        for i, addr in enumerate(vaddr):
            ndarray[addr, i] = 1.0
        # Scale the ndarray to fit this observation shape.
        return self.from_ndarray(ndarray)

    @staticmethod
    def _shrink_addresses(from_array, to_array):
        from_shape = from_array.shape()
        to_shape = to_array.shape

        p_i = from_shape[0] / to_shape[0]
        q_i = from_shape[0] - p_i * to_shape[0]

        for i in range(q_i):
            slice_i = slice(i * (p_i + 1), (i + 1) * (p_i + 1))
            to_array[i, :] = np.sum(from_array[slice_i, :], 0)


        offset = q_i * (p_i + 1)
        for i in range(q_i, to_shape[0] - q_i):
            slice_i = slice(offset + p_i * i, offset + p_i * (i + 1))
            to_array[i, :] = np.sum(from_array[slice_i, :], 0)

    @staticmethod
    def _shrink_samples(from_array, to_array):
        from_shape = from_array.shape()
        to_shape = to_array.shape

        p_j = from_shape[1] / to_shape[1]
        q_j = from_shape[1] - p_j * to_shape[1]

        for j in range(q_j):
            slice_j = slice(j * (p_j + 1), (j + 1) * (p_j + 1))
            to_array[:, j] = np.sum(from_array[:, slice_j], 0)

        offset = q_j * (p_j + 1)
        for j in range(q_j, to_shape[1] - q_j):
            slice_j = slice(offset + p_j * j, offset + p_j * (j + 1))
            to_array[:, j] = np.sum(from_array[:, slice_j], 0)

    @staticmethod
    def _grow_addresses(from_array, to_array):
        from_shape = from_array.shape()
        to_shape = to_array.shape

        p_i = to_shape[0] / from_shape[0]
        q_i = to_shape[0] - p_i * from_shape[0]

        for i in range(q_i):
            slice_i = slice(i * (p_i + 1), (i + 1) * (p_i + 1))
            to_array[slice_i, :] = from_array[i, :] / slen(slice_i)

        offset = q_i * (p_i + 1)
        for i in range(q_i, from_shape[0] - q_i):
            slice_i = slice(offset + p_i * i, offset + p_i * (i + 1))
            to_array[slice_i, :] = from_array[i, :] / slen(slice_i)

    @staticmethod
    def _grow_samples(from_array, to_array):
        from_shape = from_array.shape()
        to_shape = to_array.shape

        p_j = to_shape[1] / from_shape[1]
        q_j = to_shape[1] - p_j * from_shape[1]

        for j in range(q_j):
            slice_j = slice(j * (p_j + 1), (j + 1) * (p_j + 1))
            to_array[:, slice_j] = from_array[:, j] / slen(slice_j)

        offset = q_j * (p_j + 1)
        for j in range(q_j, from_shape[1] - q_j):
            slice_j = slice(offset + p_j * j, offset + p_j * (j + 1))
            to_array[:, slice_j] = from_array[:, j] / slen(slice_j)

    def sample(self) -> np.ndarray:
        return self.np_random.sample(self.shape(), dtype=self.dtype)

    def contains(self, x) -> bool:
        if isinstance(x, TraceSet):
            x = self.copy().from_window(x)._ndarray
        elif isinstance(x, WindowObservation):
            # Scale x to the right size.
            if x.shape() != self.shape():
                x = self.copy().from_ndarray(x._ndarray)
            x = x._ndarray
        elif isinstance(x, np.ndarray):
            x = self.copy().from_ndarray(x)._ndarray
        else:
            raise ValueError("Invalid observation type {}.".format(type(x)))
        return all(x <= self._ndarray)

## TODO test this.
