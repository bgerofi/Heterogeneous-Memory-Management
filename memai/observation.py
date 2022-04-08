import numpy as np
import unittest
from scipy.ndimage import zoom
from gym.spaces import Space

"""
This module defines the observation space for the RL algorithm and
the method to convert timestamped memory access into an observation.
"""


def slen(s: slice):
    return s.stop - s.start


class WindowObservationSpace(Space):
    """
    This class defines an observation that is a set of timestamped memory
    access inside a trace window and inside range of addresses.
    The observation it creates are a matrix of floats representing a
    fixed size image obtained by zooming in or out the matrix of timestamped
    memory accesses.
    """

    def __init__(self, nrows, ncols, seed=None):
        if nrows <= 0 or ncols <= 0:
            raise ValueError("Invalid input. nrows > 0 and ncols > 0.")

        shape = [nrows, ncols]
        super().__init__(shape, np.float32, seed)

    def __eq__(self, other):
        if not isinstance(other, WindowObservationSpace):
            return False
        if other.shape != self.shape:
            return False
        return True

    def empty(self):
        """
        Return the empty observation (for an empty window).
        """
        return np.zeros(self.shape, dtype=self.dtype)

    def from_sparse_matrix(self, x, y):
        """
        Read a sparce matrix where `x` represents timestamps and `y`
        represents addresses and output a scaled observation matrix.

        `x` and `y` are expected to be lists of positive integers and
        are scaled such that their minimum value is 1.0.

        A first empty observation matrix is created containing respectively
        as many rows and columns as the maximum scaled `x` value and the
        maximum scaled `y` value.
        For each `(x,y)` pair, the corresponding cell of the matrix is
        incremented by one.

        The resulting matrix is then zoomed in/out to fit the expected
        observation shape and returned.
        """
        x_min = min(i for i in x if i > 0)
        y_min = min(i for i in y if i > 0)

        x = (np.array(x) / x_min).astype(np.uint64)
        y = (np.array(y) / y_min).astype(np.uint64)

        x_max = int(np.nanmax(x))
        y_max = int(np.nanmax(y))

        img = np.zeros((x_max + 1, y_max + 1), dtype=self.dtype)
        for i, j in zip(x, y):
            img[i, j] += 1.0

        zoom_x = self.shape[0] / img.shape[0]
        zoom_y = self.shape[1] / img.shape[1]
        return zoom(img, (zoom_x, zoom_y))

    def sample(self) -> np.ndarray:
        """
        Create a random observation.
        """
        return self.np_random.sample(self.shape, dtype=self.dtype)

    def contains(self, x) -> bool:
        if isinstance(x, TraceSet):
            return True
        elif isinstance(x, np.ndarray):
            return len(x.shape) == 2
        else:
            return False


class TestObservationSpace(unittest.TestCase):
    def mat_to_coord(m):
        x = []
        y = []
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] != 0:
                    x.append(i)
                    y.append(j)
        return x, y

    def test_scale_down(self):
        observation_space = WindowObservationSpace(2, 2)
        input_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        check_mat = np.array([
            [ 1.0000000e+00, -1.3594064e-16],
            [-8.0429491e-17, 1.0000000e+00]
        ])
        
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))

        ##############################

        input_mat = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        check_mat = np.array([
            [2.7061686e-16, 1.0000000e+00],
            [1.0000000e+00, 2.7061686e-16]
        ])
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))

        ##############################

        observation_space = WindowObservationSpace(4, 4)
        input_mat = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ]
        )
        check_mat = np.array([
            [ 6.9657402e-19, -9.5693776e-03, -9.1440722e-02, 1.0000000e+00],
            [-9.5693776e-03, -1.8013425e-01, 8.2039773e-01, -9.1440722e-02],
            [-9.1440722e-02, 8.2039773e-01, -1.8013425e-01, -9.5693776e-03],
            [ 1.0000000e+00, -9.1440722e-02, -9.5693776e-03, -6.2423197e-18],
        ])
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))

        ##############################

        input_mat = np.array(
            [
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
            ]
        )
        check_mat = np.array([
            [-1.6077272e-18, 4.7116960e-18, 2.7458402e-16, 1.0000000e+00],
            [ 6.8084190e-18, 2.4550636e-16, 1.0000000e+00, 1.5679527e-16],
            [ 3.6745573e-16, 1.0000000e+00, -7.4507003e-18, 1.7701079e-17],
            [ 1.0000000e+00, 2.2804965e-16, -5.4224080e-18, 3.5964432e-18],
        ])
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))

    def test_scale_up(self):
        observation_space = WindowObservationSpace(4, 4)
        input_mat = np.array([
            [0, 1],
            [1, 0]
        ])
        check_mat = np.array([
            [9.7144515e-17, 2.5925925e-01, 7.4074072e-01, 1.0000000e+00],
            [2.5925925e-01, 3.8408780e-01, 6.1591220e-01, 7.4074072e-01],
            [7.4074072e-01, 6.1591220e-01, 3.8408780e-01, 2.5925925e-01],
            [1.0000000e+00, 7.4074072e-01, 2.5925925e-01, 9.7144515e-17],
        ])
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))

        ##############################

        input_mat = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        check_mat = np.array([
            [ 2.7061686e-16, -1.1111111e-01, 3.7037036e-01, 1.0000000e+00],
            [-1.1111111e-01, 4.6639231e-01, 6.9821674e-01, 3.7037036e-01],
            [ 3.7037036e-01, 6.9821674e-01, 4.6639231e-01, -1.1111111e-01],
            [ 1.0000000e+00, 3.7037036e-01, -1.1111111e-01, 2.7061686e-16],
        ])
        output_mat = observation_space.from_sparse_matrix(
            *TestObservationSpace.mat_to_coord(input_mat)
        )
        self.assertTrue(all(check_mat.flatten() - output_mat.flatten() < 1e-8))


if __name__ == "__main__":
    unittest.main()
