from gym.spaces import Space
import numpy as np
from intervaltree import IntervalTree, Interval


class DefaultActionSpace(Space):
    def __init__(
        self,
        num_actions,
        move_page_cost=10,
        page_size=1 << 14,
        seed=None,
    ):
        super().__init__([num_actions], np.uint8, seed)
        self._move_page_cost = move_page_cost
        self._page_size = page_size
        self._page_shift = int(np.round(np.log2(1 << 14)))
        self._page_mask = ~(page_size - 1)

    def do_action(self, action, memory, *args):
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

    def do_action(self, actions, memory, *args):
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
        rm_pages = [
            Interval(p, p + chunk_size) for a, p in zip(actions, pages) if not a
        ]
        add_pages = [Interval(p, p + chunk_size) for a, p in zip(actions, pages) if a]

        free_size = memory.free(rm_pages)
        alloc_size = memory.alloc(add_pages)

        n_free_pages = free_size >> self._page_shift
        n_alloc_pages = alloc_size >> self._page_shift

        return self._move_page_cost * (n_free_pages + n_alloc_pages)
