from gym.spaces import Space
import numpy as np


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

        begin_index = np.nanmin(pages) >> self._page_shift
        end_index = np.nanmax(pages) >> self._page_shift
        index_range = max(end_index - begin_index, len(actions))
        begin_index, end_index = NeighborActionSpace.extend_and_center(
            begin_index, end_index, 3 * index_range
        )

        begin_page = begin_index << self._page_shift
        end_page = end_index << self._page_shift
        chunk_size = ((end_page - begin_page) // len(actions)) & self._page_mask
        pages = range(begin_page, end_page + self._page_size, chunk_size)

        add_pages = np.unique([p for a, p in zip(actions, pages) if a])
        add_pages = [p for p in add_pages if p not in hbm_intervals]

        remove_pages = np.unique([p for a, p in zip(actions, pages) if not a])
        remove_pages = [p for p in remove_pages if p in hbm_intervals]

        for p in remove_pages:
            hbm_intervals.removei(p, p + chunk_size)

        available_size = self._hbm_size - sum([i.end - i.begin for i in hbm_intervals])
        add_pages = add_pages[: available_size // chunk_size]

        for p in add_pages:
            hbm_intervals.addi(p, p + chunk_size)
        hbm_intervals.merge_overlaps(strict=False)

        return (len(add_pages) + len(remove_pages)) * self._move_page_cost
