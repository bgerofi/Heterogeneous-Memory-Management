import numpy as np
from intervaltree import IntervalTree


def interval_tree_size(intervals):
    return sum((i.end - i.begin for i in intervals))


class Memory:
    def __init__(
        self, capacity=np.iinfo(np.int64).max, merge_overlap_threshold=1 << 12
    ):
        self.capacity = capacity
        self._merge_overlap_threshold = merge_overlap_threshold
        self._chunks = IntervalTree()
        self.alloc_size = 0
        self.free_size = 0
        self.double_alloc_size = 0
        self.double_free_size = 0
        self.capacity_overflow_size = 0

    def report(self):
        s = "Chunks Moves Summary:\n"
        s += "\tTotal Successful Allocation: {:.1f} (MB)\n".format(
            self.alloc_size / 1e6
        )
        s += "\tTotal Successful Free: {:.1f} (MB)\n".format(self.free_size / 1e6)
        s += "\tDouble Allocations Size: {:.1f} (MB)\n".format(
            self.double_alloc_size / 1e6
        )
        s += "\tCapacity Overflow Allocations Size: {:.1f} (MB)\n".format(
            self.capacity_overflow_size / 1e6
        )
        s += "\tDouble Free Size: {:.1f} (MB)\n".format(self.double_free_size / 1e6)
        return s

    def empty(self):
        self._chunks = IntervalTree()
        self.alloc_size = 0
        self.free_size = 0
        self.double_alloc_size = 0
        self.double_free_size = 0
        self.capacity_overflow_size = 0

    @property
    def occupancy(self):
        return interval_tree_size(self._chunks)

    def __contains__(self, item):
        return item in self._chunks

    def alloc(self, address_intervals, report=False):
        # Create an interval tree of adjacent chunks to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)
        if report:
            total_size = interval_tree_size(data)
        else:
            total_size = 0

        # Trim chunks to allocate to remove those already allocated.
        data.difference_update(self._chunks)
        if report:
            new_size = interval_tree_size(data)
        else:
            new_size = 0

        # Compute remaining size memory and add only intervals that fits in it.
        if len(data) > 0:
            remaining_size = self.capacity - self.occupancy
            add_size = np.cumsum([i.end - i.begin for i in data])
            stop_index = next((i for i, s in enumerate(data) if s > remaining_size), -1)
            if stop_index > 0:
                data = [i for i in data][:stop_index]
                self._chunks.update(data)
            elif stop_index == -1:
                self._chunks.update(data)
            else:
                data = []
        fit_size = interval_tree_size(data)

        # Critical optimization here. 1<<12 has been chosen empirically.
        if len(data) > 0 and len(self._chunks) > self._merge_overlap_threshold:
            self._chunks(strict=False)

        self.alloc_size += fit_size
        self.capacity_overflow_size += new_size - fit_size
        self.double_alloc_size += total_size - new_size

        return fit_size

    def free(self, address_intervals, report=False):
        # Create an interval tree of adjacent chunks to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        if report:
            total_size = interval_tree_size(data)
        else:
            total_size = 0

        # Trim chunks to remove to avoid acounting for those already not allocated.
        data.intersection_update(self._chunks)
        removed_size = interval_tree_size(data)

        self._chunks.difference_update(data)

        self.double_free_size += total_size - removed_size
        self.free_size += removed_size

        return removed_size
