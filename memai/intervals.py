import numpy as np
from intervaltree import IntervalTree


class AddressSpace:
    def __init__(self, intervals, page_size=1 << 14):
        self._index_addr_ = []
        self._addr_index_ = {}
        self._page_mask_ = ~(page_size - 1)
        self.page_size = page_size

        i = 0
        for interval in intervals:
            low = interval.begin & self._page_mask_
            high = interval.end & self._page_mask_
            high = high + page_size if high != interval.end else high
            self.index_addr += list(range(low, high, page_size))
            self._addr_index_.update(
                {addr: i + j for j, addr in enumerate(range(low, high, page_size))}
            )
            i += (high - low) / page_size

    def addr_index(self, addr):
        try:
            return self._addr_index_[addr & self._page_mask_]
        except KeyError:
            raise ValueError("Invalid address not in address space.")

    def index_addr(self, i):
        return self._index_addr_[i]

    def __len__(self):
        return len(self._index_address_)

    def __getitem__(self, addr):
        return self.addr_index(addr)


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
