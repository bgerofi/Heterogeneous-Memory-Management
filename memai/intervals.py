import numpy as np
from intervaltree import IntervalTree


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self.intervals = IntervalTree()
        self._page_size_ = page_size
        self._page_mask_ = ~(page_size - 1)
        self._interval_distance_ = interval_distance * page_size

    def append_addresses(self, vaddr):
        low_vaddr, high_vaddr = self._page_bounds_(vaddr)
        try:
            for low, high in zip(low_vaddr, high_vaddr):
                self.intervals[low:high] = 1
        except TypeError:
            self.intervals[low_vaddr:high_vaddr] = 1

        self.intervals = IntervalDetector._identified_intervals_(self.intervals)
        return self

    def _page_bounds_(self, vaddr):
        if isinstance(vaddr, (int, np.int64, np.int32, np.uint64, np.uint32)):
            low = (vaddr & self._page_mask_) - self._interval_distance_
            high = (vaddr + self._interval_distance_) & self._page_mask_
            return max(0, low), high

        if isinstance(vaddr, list):
            vaddr = np.array(vaddr)
        if isinstance(vaddr, (np.ndarray, np.array)):
            vaddr = vaddr.reshape(1, len(b))

        low = (vaddr & self._page_mask_) - self._interval_distance_
        low = np.apply_along_axis(lambda i: max(i, 0), 0, low)
        high = (vaddr + self._interval_distance_) & self._page_mask_
        return low, high

    @staticmethod
    def _identified_intervals_(intervals):
        intervals.merge_overlaps()
        new_intervals = IntervalTree()
        for i, interval in enumerate(intervals):
            new_intervals.add(Interval(interval.begin, interval.end, 0))
        return new_intervals

    @staticmethod
    def from_intervals(intervals, page_size=1 << 14, interval_distance=1 << 22):
        i_detector = IntervalDetector.__new__()
        i_detector.intervals = IntervalTree()
        i_detector._page_size_ = page_size
        i_detector._page_mask_ = ~(page_size - 1)
        i_detector._interval_distance_ = interval_distance * page_size

        begin = [i.begin for i in intervals()]
        end = [i.end for i in intervals()]

        begin, _ = i_detector._page_bounds_(begin)
        _, end = i_detector._page_bounds_(end)

        i_detector.intervals = IntervalTree(
            [Interval(low, high)] for low, high in zip(begin, end)
        )
        i_detector.intervals = IntervalDetector._identified_intervals_(i.intervals)
        return i_detector

    def num_intervals(self):
        return len(self.intervals)

    def interval_index(self, vaddr):
        return self.intervals[vaddr].data

    def interval_at(self, index):
        return next((it for i, it in enumerate(self.intervals) if i == index), None)

    def interval_len(self, interval):
        return (interval.end - interval.begin) / self._page_size_

    def max_interval_len(self):
        return np.nanmax([self.interval_len(i) for i in self.intervals])

    def interval_addr_index(self, addr):
        interval = self.intervals[addr]
        addr = addr & self._page_mask_
        return (addr - interval.begin) / self._page_size_

    def interval_addr_at(self, interval, index):
        return interval.begin + index * self._page_size_

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def __getitem__(self, addr):
        return self.intervals[addr]
