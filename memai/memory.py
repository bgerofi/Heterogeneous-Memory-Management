import numpy as np
from intervaltree import IntervalTree, Interval
from functools import reduce
import unittest


class Memory:
    def __init__(
        self, capacity=np.iinfo(np.int64).max, merge_overlap_threshold=1 << 12
    ):
        self.capacity = capacity
        self._merge_overlap_threshold = merge_overlap_threshold
        self.empty()

    def empty(self):
        self._chunks = IntervalTree()
        self.alloc_size = 0
        self.free_size = 0
        self.double_alloc_size = 0
        self.double_free_size = 0
        self.capacity_overflow_size = 0

    @property
    def occupancy(self):
        return sum(i.length() for i in self._chunks)

    def __contains__(self, item):
        return item in self._chunks

    def _update_(self, intervals):
        self._chunks.update(intervals)
        if len(self._chunks) > self._merge_overlap_threshold:
            self._chunks.merge_overlaps(strict=False)

    def report(self):
        s = "Allocation/Free Summary:\n"
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

    def fast_alloc(self, address_intervals):
        """
        Faster allocation method.
        This method only loosely prevent capacity overflow.
        Some allocations not overflowing may fail.
        This method also does not keep track of double allocations, allocated size an overflowing size.
        """
        # Merge overlaps of addresses to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        available_size = self.capacity - sum(i.length() for i in self._chunks)
        data = sorted(data, key=lambda i: i.length())
        sizes = np.cumsum([i.length() for i in data])

        if sizes[-1] < available_size:
            self._update_(data)
            return sizes[-1]

        cut = next(i for i, s in enumerate(sizes) if s > available_size)
        data = data[: cut + 1]
        data[-1] = Interval(
            data[-1].begin, data[-1].begin + (available_size - sizes[cut - 1])
        )
        self._update_(data)
        return available_size

    def alloc(self, address_intervals):
        """
        This is an exact allocation method.
        Input interval will be allocated as follow.
        If some address ranges overlap already allocated address ranges, they
        are not allocated and will be counted as a double allocation.
        If their is too much data compared to capacity, the exceeding size
        will be counted and the data which will be allocated is the first non
        already allocated ranges (sorted by address) that fit in the memory.
        """

        # Merge overlaps of addresses to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        non_overlapping = self._chunks.copy()
        non_overlapping.update(data)
        non_overlapping.split_overlaps()
        for i in self._chunks:
            non_overlapping.remove_envelop(i.begin, i.end)

        available_size = self.capacity - sum(i.length() for i in self._chunks)
        total_size = sum(i.length() for i in data)
        alloc_size = sum(i.length() for i in non_overlapping)
        overlap_size = total_size - alloc_size
        final_alloc_size = min(alloc_size, available_size)
        overflow_size = abs(final_alloc_size - alloc_size)

        self.alloc_size += final_alloc_size
        self.double_alloc_size += overlap_size
        self.capacity_overflow_size += overflow_size

        if available_size <= 0:
            return 0

        if overflow_size == 0:
            self._chunks.update(data)
            self._chunks.merge_overlaps(strict=False)
            return final_alloc_size

        non_overlapping = sorted(non_overlapping, key=lambda i: i.length())
        sizes = np.cumsum([i.length() for i in non_overlapping])
        cut = next(i for i, s in enumerate(sizes) if s > available_size)
        non_overlapping = non_overlapping[: cut + 1]
        length = available_size if cut == 0 else available_size - sizes[cut - 1]
        non_overlapping[-1] = Interval(
            non_overlapping[-1].begin,
            non_overlapping[-1].begin + length,
        )

        self._update_(IntervalTree(non_overlapping))
        return final_alloc_size

    def free(self, address_intervals):
        # Create an interval tree of adjacent chunks to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        free_size = 0
        non_free_size = 0
        for new_chunk in data:
            overlapping_intervals = sorted(self._chunks.overlap(new_chunk))

            if len(overlapping_intervals) == 0:
                non_free_size += new_chunk.length()
                continue

            for i in range(1, len(overlapping_intervals)):
                free_size += overlapping_intervals[i].length()
                non_free_size += (
                    overlapping_intervals[i].begin - overlapping_intervals[i - 1].end
                )
                self._chunks.remove(overlapping_intervals[i])
            self._chunks.remove(overlapping_intervals[0])
            free_size += overlapping_intervals[0].length()

            if new_chunk.begin <= overlapping_intervals[0].begin:
                non_free_size += overlapping_intervals[0].begin - new_chunk.begin
            else:
                free_size -= overlapping_intervals[0].length()
                free_size += overlapping_intervals[0].end - new_chunk.begin
                self._chunks.add(
                    Interval(overlapping_intervals[0].begin, new_chunk.begin)
                )
            if new_chunk.end >= overlapping_intervals[-1].end:
                non_free_size += new_chunk.end - overlapping_intervals[-1].end
            else:
                free_size -= overlapping_intervals[-1].length()
                free_size += new_chunk.end - overlapping_intervals[-1].begin
                self._chunks.add(Interval(new_chunk.end, overlapping_intervals[-1].end))

        self.double_free_size += non_free_size
        self.free_size += free_size
        return free_size


class TestMemory(unittest.TestCase):
    def test_alloc(self):
        mem = Memory(10)
        self.assertEqual(mem.occupancy, 0)

        self.assertEqual(mem.alloc([Interval(0, 1)]), 1)
        self.assertEqual(mem.occupancy, 1)

        self.assertEqual(mem.alloc([Interval(0, 1)]), 0)
        self.assertEqual(mem.occupancy, 1)
        self.assertEqual(mem.double_alloc_size, 1)

        self.assertEqual(mem.alloc([Interval(0, 2)]), 1)
        self.assertEqual(mem.occupancy, 2)
        self.assertEqual(mem.double_alloc_size, 2)

        self.assertEqual(mem.alloc([Interval(3, 4)]), 1)
        self.assertEqual(mem.occupancy, 3)

        self.assertEqual(mem.alloc([Interval(2, 11)]), 7)
        self.assertEqual(mem.double_alloc_size, 3)
        self.assertEqual(mem.capacity_overflow_size, 1)
        self.assertEqual(mem.occupancy, 10)

    def test_free(self):
        mem = Memory(10)
        mem.alloc([Interval(0, 10)])

        self.assertEqual(mem.free([Interval(10, 11)]), 0)
        self.assertEqual(mem.free_size, 0)
        self.assertEqual(mem.double_free_size, 1)

        self.assertEqual(mem.free([Interval(4, 5)]), 1)
        self.assertEqual(mem.free_size, 1)
        self.assertEqual(mem.double_free_size, 1)

        self.assertEqual(mem.free([Interval(3, 6)]), 2)
        self.assertEqual(mem.free_size, 3)
        self.assertEqual(mem.double_free_size, 2)

        self.assertEqual(mem.free([Interval(8, 11)]), 2)
        self.assertEqual(mem.free_size, 5)
        self.assertEqual(mem.double_free_size, 3)


if __name__ == "__main__":
    unittest.main()
