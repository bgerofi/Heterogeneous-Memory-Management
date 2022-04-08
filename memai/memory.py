import numpy as np
from intervaltree import IntervalTree, Interval
from functools import reduce
import unittest

"""
This module implements tracking of data present of absent from a memory.
`alloc()` and `free()` method will add or remove data and update statistic
on data that could not be added because the memory capacity is exceeded or
because it is already in the memory, and data that could not be removed because
it was not there.
"""


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
        """
        This is a method to avoid calling `IntervalTree.merge_overlaps()` too
        often because it significantly slows down the allocation.
        """
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

    def alloc(self, address_intervals):
        """
        This is an exact allocation method.
        Input interval will be allocated as follow.
        If some address ranges overlap already allocated address ranges, the
        overlapping part will not be allocated and will be counted as a
        double allocation.
        If their is too much data compared to the memory capacity, the
        exceeding size is recorded and the data which will be allocated
        is the first non already allocated ranges (sorted by size) that fit
        in the memory.
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
            if overlap_size > 0:
                self._chunks.update(data)
                self._chunks.merge_overlaps(strict=False)
            else:
                self._update_(data)
            return final_alloc_size

        # The memory capacity is exceeded. We have to trim the data to
        # allocate to allocate only what fits.
        non_overlapping = sorted(non_overlapping, key=lambda i: i.length())
        sizes = np.cumsum([i.length() for i in non_overlapping])
        cut = next(i for i, s in enumerate(sizes) if s > available_size)
        non_overlapping = non_overlapping[: cut + 1]
        length = available_size if cut == 0 else available_size - sizes[cut - 1]

        if length > 0:
            non_overlapping[-1] = Interval(
                non_overlapping[-1].begin,
                non_overlapping[-1].begin + length,
            )
        else:
            non_overlapping = non_overlapping[:-1]

        if len(non_overlapping) > 0:
            self._update_(IntervalTree(non_overlapping))
        return final_alloc_size

    def free(self, address_intervals):
        """
        Remove ranges of addresses from the memory.
        If addresses are not present in the memory, they are recorded
        as double free, otherwise they are removed and recorded as freed
        memory.
        """

        # Create an interval tree of adjacent chunks to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        free_size = 0
        non_free_size = 0

        # Iterate over data to free,
        # For each chunk get overlapping allocated memory regions.
        # Count non overlapping left and right regions as double free
        # and everything else as a free.
        for new_chunk in data:
            overlapping_intervals = sorted(self._chunks.overlap(new_chunk))

            # If nothing overlaps, it is only double frees.
            if len(overlapping_intervals) == 0:
                non_free_size += new_chunk.length()
                continue

            # Remove every overlapping interval.
            # Count the interval length as a free and the space in between
            # intervals as double free.
            for i in range(1, len(overlapping_intervals)):
                free_size += overlapping_intervals[i].length()
                non_free_size += (
                    overlapping_intervals[i].begin - overlapping_intervals[i - 1].end
                )
                self._chunks.remove(overlapping_intervals[i])
            self._chunks.remove(overlapping_intervals[0])
            free_size += overlapping_intervals[0].length()

            # Count non overlapping interval on the left if interval to free is
            # larger than allocated interval.
            # Else if interval to free is smaller, we have to reinsert the
            # non-freed part.
            if new_chunk.begin <= overlapping_intervals[0].begin:
                non_free_size += overlapping_intervals[0].begin - new_chunk.begin
            else:
                free_size -= overlapping_intervals[0].length()
                free_size += overlapping_intervals[0].end - new_chunk.begin
                self._chunks.add(
                    Interval(overlapping_intervals[0].begin, new_chunk.begin)
                )

            # Count non overlapping interval on the right if interval to free
            # is larger than allocated interval.
            # Else if interval to free is smaller, we have to reinsert the
            # non-freed part.
            if new_chunk.end >= overlapping_intervals[-1].end:
                non_free_size += new_chunk.end - overlapping_intervals[-1].end
            else:
                free_size -= overlapping_intervals[-1].length()
                free_size += new_chunk.end - overlapping_intervals[-1].begin
                self._chunks.add(Interval(new_chunk.end, overlapping_intervals[-1].end))

        self.double_free_size += non_free_size
        self.free_size += free_size
        return free_size


class FastMemory(Memory):
    """
    Same as memai.memory.Memory but with faster allocation method.
    This memory allocation method only loosely prevent capacity overflow.
    Some allocations not overflowing may fail.
    This method also does not keep track of double allocations, allocated
    size an overflowing size.
    """

    def alloc(self, address_intervals):
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
        length = available_size if cut == 0 else available_size - sizes[cut - 1]
        if length > 0:
            data[-1] = Interval(data[-1].begin, data[-1].begin + length)
        else:
            data = data[:-1]

        if len(data) > 0:
            self._update_(IntervalTree(data))
            return available_size
        else:
            return 0


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
