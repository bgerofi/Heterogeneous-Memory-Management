from intervaltree import IntervalTree


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self.intervals = IntervalTree()
        self._page_size_ = page_size
        self._page_mask_ = ~(page_size - 1)
        self._interval_distance_ = interval_distance * page_size

    def append_addr(self, vaddr):
        low_vaddr = (vaddr & self._page_mask_) - self._interval_distance_
        high_vaddr = (vaddr + self._interval_distance_) & self._page_mask_
        try:
            low_vaddr = (max(0, addr) for addr in low_vaddr)
            for low, high in zip(low_vaddr, high_vaddr):
                self.intervals[low:high] = 1
        except TypeError:
            low_vaddr = max(0, low_vaddr)
            self.intervals[low_vaddr:high_vaddr] = 1

        self.intervals.merge_overlaps(data_reducer=lambda a, b: a + b)
        return self

    def offsetof(self, vaddr):
        try:
            return (i - next(iter(self.intervals[i])).begin for i in vaddr)
        except TypeError:
            return vaddr - next(iter(self.intervals[vaddr])).begin
