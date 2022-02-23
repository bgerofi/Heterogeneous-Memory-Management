import numpy as np
from intervaltree import IntervalTree


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self.intervals = IntervalTree()
        self._page_mask_ = ~(page_size - 1)
        self._interval_distance_ = interval_distance * page_size

    def append_addresses(self, vaddr):
        low_vaddr, high_vaddr = self._page_bounds_(vaddr)
        for low, high in zip(low_vaddr, high_vaddr):
            self.intervals[low:high] = 0
        self.intervals.merge_overlaps()
        return self

    def _page_bounds_(self, vaddr):
        if isinstance(vaddr, list):
            vaddr = np.array(vaddr)
        if isinstance(vaddr, (np.ndarray, np.array)):
            vaddr = vaddr.reshape(1, len(b))
        low = (vaddr & self._page_mask_) - self._interval_distance_
        low = np.apply_along_axis(lambda i: max(i, 0), 0, low)
        high = (vaddr + self._interval_distance_) & self._page_mask_
        return low, high
