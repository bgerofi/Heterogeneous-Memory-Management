import numpy as np
from intervaltree import IntervalTree, Interval
import unittest


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

    def alloc(self, address_intervals):
        # Merge overlaps of addresses to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        alloc_size = 0
        non_alloc_size = 0
        overflow_size = 0
        allocated_size = sum(i.length() for i in self._chunks)
        available_size = self.capacity - allocated_size

        for new_chunk in data:
            if available_size <= 0:
                return alloc_size
            overlapping_intervals = sorted(self._chunks.overlap(new_chunk))

            if len(overlapping_intervals) == 0:
                end = min(new_chunk.end, new_chunk.begin + available_size)
                chunk = Interval(new_chunk.begin, end)
                alloc_size += end - chunk.begin
                overflow_size += new_chunk.end - end
                available_size -= end - chunk.begin
                self._chunks.add(chunk)
                continue

            non_alloc_size += sum(i.length() for i in overlapping_intervals)

            if new_chunk.begin < overlapping_intervals[0].begin:
                end = min(
                    overlapping_intervals[0].begin, new_chunk.begin + available_size
                )
                chunk = Interval(new_chunk.begin, end)
                size = end - chunk.begin
                alloc_size += size
                available_size -= size
                overflow_size += new_chunk.end - chunk.end
                non_alloc_size -= overlapping_intervals[0].end - new_chunk.end
                self._chunks.add(chunk)

            if overlapping_intervals[-1].end < new_chunk.end:
                end = min(new_chunk.end, overlapping_intervals[-1].end + available_size)
                chunk = Interval(overlapping_intervals[-1].end, end)
                size = end - chunk.begin
                overflow_size += new_chunk.end - chunk.end
                non_alloc_size -= new_chunk.begin - overlapping_intervals[-1].begin
                alloc_size += size
                available_size -= size
                self._chunks.add(chunk)

        self._chunks.merge_overlaps(strict=False)
        self.alloc_size += alloc_size
        self.double_alloc_size += non_alloc_size
        self.capacity_overflow_size += overflow_size
        return alloc_size

    def free(self, address_intervals):
        # Create an interval tree of adjacent chunks to allocate.
        data = IntervalTree(address_intervals)
        data.merge_overlaps(strict=False)

        free_size = 0
        non_free_size = 0
        for new_chunk in data:
            overlapping_intervals = sorted(self._chunks.overlap(new_chunk))
            for i in overlapping_intervals:
                free_size += i.length()
                self._chunks.remove(i.begin, i.end)

            if overlapping_intervals[0].begin < new_chunk.begin:
                size = new_chunk.begin - overlapping_intervals[0].begin
                free_size -= size
                non_free_size += size
                self._chunks.add(
                    Interval(overlapping_intervals[0].begin, new_chunk.begin)
                )
            if overlapping_intervals[-1].end > new_chunk.end:
                size = overlapping_intervals[-1].end - new_chunk.end
                free_size -= size
                non_free_size += size
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

        self.assertEqual(mem.alloc([Interval(2, 3)]), 1)
        self.assertEqual(mem.occupancy, 3)

        self.assertEqual(mem.alloc([Interval(2, 11)]), 7)
        self.assertEqual(mem.occupancy, 10)
        self.assertEqual(mem.double_alloc_size, 3)
        self.assertEqual(mem.capacity_overflow_size, 1)

    def test_free(self):
        pass

    def test_combined(self):
        pass


if __name__ == "__main__":
    unittest.main()
