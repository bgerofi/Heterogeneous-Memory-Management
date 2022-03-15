from gym.spaces import Space
import numpy as np
from intervaltree import IntervalTree, Interval


class DefaultActionSpace(Space):
    def __init__(
        self,
        num_actions,
        move_page_cost=10,
        page_size=1 << 14,
        hbm_size=np.iinfo(np.int64).max,
        seed=None,
    ):
        super().__init__([num_actions], np.uint8, seed)
        self._move_page_cost = move_page_cost
        self._page_size = page_size
        self._page_shift = int(np.round(np.log2(1 << 14)))
        self._page_mask = ~(page_size - 1)
        self._hbm_size = hbm_size

    def remaining_size(self, hbm_intervals):
        return self._hbm_size - sum([i.end - i.begin for i in hbm_intervals])

    def do_action(self, action, hbm_intervals, *args):
        return 0

    def all_to_hbm_action(self):
        return np.ones(self.shape[0], dtype=np.uint8)

    def all_to_ddr_action(self):
        return np.zeros(self.shape[0], dtype=np.uint8)

    def sample(self) -> np.ndarray:
        return self.np_random.randint(0, 2, self._shape, dtype=self.dtype)

    def contains(self, x):
        if isinstance(x, np.ndarray):
            return x.shape == self.shape
        if isinstance(x, list):
            return len(x) == self.shape[0]
        return False


class NeighborActionSpace(DefaultActionSpace):
    @staticmethod
    def extend_and_center(begin, end, size):
        extra = (end - begin) + size
        extra_left = extra // 2
        extra_right = extra - extra_left

        begin = 0 if begin < extra_left else (begin - extra_left)
        new_end = end + extra_right
        end = np.iinfo(type(end)).max if new_end < end else new_end

        return begin, end

    def do_action(self, actions, hbm_intervals, *args):
        observation, pages, count, t_ddr, t_hbm = args
        if len(pages) == 0:
            return 0

        # Compute a range of chunk indexes of size: 3 * len(actions)
        # around and centered on accessed pages.
        begin_index = np.nanmin(pages) >> self._page_shift
        end_index = np.nanmax(pages) >> self._page_shift
        # Make sure there is at least len(actions) chunks.
        index_range = max(end_index - begin_index, len(actions))
        # Multiply by 3 for chunks before, into and after the range of accessed
        # pages.
        begin_index, end_index = NeighborActionSpace.extend_and_center(
            begin_index, end_index, 3 * index_range
        )
        # Translate indexes back into pages address.
        begin_page = begin_index << self._page_shift
        end_page = end_index << self._page_shift
        # Compute the size of each chunk from begin_page to end_page
        # such that there is one chunk per action.
        # Also make chunk_size a multiple of page_size to be sure to get addresses
        # on a page boundary.
        chunk_size = ((end_page - begin_page) // len(actions)) & self._page_mask
        pages = range(begin_page, end_page + self._page_size, chunk_size)

        # Prepare new pages that will be added to hbm intervals. We only keep pages
        # That are not already in hbm intervals.
        add_pages = [Interval(p, p + chunk_size) for a, p in zip(actions, pages) if a]
        add_pages = IntervalTree(add_pages)
        add_pages.merge_overlaps(strict=False)
        add_pages.difference_update(hbm_intervals)

        # Remove selected pages from hbm intervals.
        rm_pages = [
            Interval(p, p + chunk_size) for a, p in zip(actions, pages) if not a
        ]
        rm_pages = IntervalTree(rm_pages)
        rm_pages.merge_overlaps(strict=False)
        rm_pages.intersection_update(hbm_intervals)
        hbm_intervals.difference_update(rm_pages)

        # Compute remaining size of hbm memory and add only intervals that fits in it.
        if len(add_pages) > 0:
            available_size = self.remaining_size(hbm_intervals)
            add_size = np.cumsum([i.end - i.begin for i in add_pages])
            stop_index = next(
                (i for i, s in enumerate(add_size) if s > available_size), -1
            )
            if stop_index > 0:
                add_pages = [i for i in add_pages][:stop_index]
                add_size = add_size[stop_index - 1]
                hbm_intervals.update(add_pages)
            elif stop_index == -1:
                add_size = add_size[-1]
                hbm_intervals.update(add_pages)
            else:
                add_pages = []
                add_size = 0

        # Critical optimization here. 1<<12 has been chosen empirically.
        if len(add_pages) > 0 and len(hbm_intervals) > (1 << 12):
            hbm_intervals.merge_overlaps(strict=False)

        add_size = sum((i.end - i.begin) for i in add_pages)
        return (len(add_pages) + len(rm_pages)) * self._move_page_cost
