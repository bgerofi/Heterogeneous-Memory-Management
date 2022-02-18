import numpy as np
from intervaltree import IntervalTree


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self.intervals = IntervalTree()
        self._page_size_ = page_size
        self._page_mask_ = ~(page_size - 1)
        self._interval_distance_ = interval_distance * page_size

    def page_bounds(self, vaddr):
        low_vaddr = (vaddr & self._page_mask_) - self._interval_distance_
        high_vaddr = (vaddr + self._interval_distance_) & self._page_mask_
        return low_vaddr, high_vaddr

    def interval_num_pages(self, interval):
        return (interval.end-interval.begin) / self._page_size_
    
    def max_interval_num_pages(self):
        return np.nanmax([ self.interval_num_pages(i) for i in self.intervals ])

    def addressof(self, indexes, interval):
        return interval.begin + self._page_size_ * np.array(indexes)

    def indexof(self, addresses):
            return ((i - next(iter(self.intervals[i])).begin) / self._page_size_ for i in vaddr)

    def append_addr(self, vaddr):
        low_vaddr, high_vaddr = self.page_bounds(vaddr)
        try:
            low_vaddr = (max(0, addr) for addr in low_vaddr)
            for low, high in zip(low_vaddr, high_vaddr):
                self.intervals[low:high] = 1
        except TypeError:
            low_vaddr = max(0, low_vaddr)
            self.intervals[low_vaddr:high_vaddr] = 1

        self.intervals.merge_overlaps(data_reducer=lambda a, b: a + b)
        return self

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)
