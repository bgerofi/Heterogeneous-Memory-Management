from gym.spaces import Space
import numpy as np


class MovePagesActionSpace(Space):
    def __init__(self, num_actions, move_page_cost=10, page_size=1 << 14, seed=None):
        super().__init__([num_actions], np.uint8, seed)
        self._move_page_cost = move_page_cost
        self._page_size = page_size
        self._page_mask = ~(page_size - 1)

    def _addr_to_page(self, addr):
        return [Interval(i, i + self._page_size) for i in addr & self._page_mask]

    def add_pages(self, addresses, hbm_intervals):
        pages = self._addr_to_page(addresses)
        pages = [p for p in pages if p not in hbm_intervals]
        hbm_intervals.add(pages)
        hbm_intervals.merge_overlaps(strict=False)
        return len(pages)

    def remove_pages(self, addresses, hbm_intervals):
        pages = self._addr_to_page(addresses)
        pages = [p for p in pages if p in hbm_intervals]
        hbm_intervals.remove(pages)
        hbm_intervals.merge_overlaps(strict=False)
        return len(pages)

    def do_action(self, action, action_addr_interval, hbm_intervals):
        return 0

    def sample(self) -> np.ndarray:
        return self.np_random.randint(0, 2, self._shape, dtype=self.dtype)

    def contains(self, x):
        if isinstance(x, np.ndarray):
            return x.shape == self.shape
        if isinstance(x, list):
            return len(x) == self.shape[0]
        return False
