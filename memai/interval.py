import unittest
import numpy as np
from intervaltree import IntervalTree


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self._intervals = IntervalTree()

        page_shift = np.log2(page_size)
        if float(int(page_shift)) != page_shift:
            raise ValueError(
                "Page size must be a power of 2. {} {} {}".format(
                    page_size, float(int(page_shift)), page_shift
                )
            )
        self._page_shift = int(page_shift)
        self._page_mask = ~(page_size - 1)

        # This is to make intervals aligned on a page boundary and to avoid computing
        # the page address when looking up intervals.
        if interval_distance < page_size or interval_distance % page_size != 0:
            raise ValueError("Interval distance must be a multiple of page size.")
        self._interval_distance_ = interval_distance

    def append_addresses(self, vaddr):
        low_vaddr, high_vaddr = self._page_bounds_(vaddr)

        if isinstance(low_vaddr, np.ndarray):
            low_vaddr = low_vaddr.flatten()
        if isinstance(high_vaddr, np.ndarray):
            high_vaddr = high_vaddr.flatten()

        for low, high in zip(low_vaddr, high_vaddr):
            self._intervals[low:high] = 0
        self._intervals.merge_overlaps(strict=False)
        return self

    def _page_bounds_(self, vaddr):
        low = (vaddr & self._page_mask) - self._interval_distance_
        low = low.reshape((1, low.size))
        low = np.apply_along_axis(lambda i: max(i, 0), 0, low)
        high = (vaddr + self._interval_distance_) & self._page_mask
        return low, high

    @property
    def intervals(self):
        return self._intervals

    def __iter__(self):
        return iter(self._intervals)

    def __len__(self):
        return len(self._intervals)


class TestIntervalDetector(unittest.TestCase):
    def test_detector(self):
        intervals = IntervalDetector(interval_distance=4, page_size=2)
        addresses = np.array([0, 8, 16, 26])
        intervals.append_addresses(addresses)

        # [ (0, 20), (22, 30) ]
        self.assertEqual(len(intervals), 2)

        intervals = iter(sorted(intervals))
        interval = next(intervals)
        self.assertEqual(interval.begin, 0)
        self.assertEqual(interval.end, 20)

        interval = next(intervals)
        self.assertEqual(interval.begin, 22)
        self.assertEqual(interval.end, 30)


if __name__ == "__main__":
    unittest.main()
