from gym.spaces import Space
import numpy as np
from intervaltree import IntervalTree, Interval
import unittest


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
    def match_actions_pages(actions, pages):
        page_min = np.nanmin(pages)
        page_max = np.nanmax(pages)
        size = page_max - page_min

        page_min = 0 if page_min <= size else page_min - size
        page_max = (
            (page_max + size)
            if (page_max + size) > page_max
            else np.iinfo(type(page_max)).max
        )
        chunk_size = max(1, (page_max - page_min) // len(actions))

        return zip(
            actions,
            ((p, p + chunk_size) for p in range(page_min, page_max, chunk_size)),
        )

    def do_action(self, actions, memory, *args):
        observation, pages, count, t_ddr, t_hbm = args
        if len(pages) == 0:
            return 0
        action_pages = NeighborActionSpace.match_actions_pages(actions, pages)
        rm_pages = [
            Interval(begin & self._page_mask, end & self._page_mask)
            for a, (begin, end) in action_pages
            if not a
        ]
        add_pages = [
            Interval(begin & self._page_mask, end & self._page_mask)
            for a, (begin, end) in actions_pages
            if a
        ]

        free_size = memory.free(rm_pages)
        alloc_size = memory.alloc(add_pages)

        n_free_pages = free_size >> self._page_shift
        n_alloc_pages = alloc_size >> self._page_shift

        return self._move_page_cost * (n_free_pages + n_alloc_pages)


class TestNeighborActionSpace(unittest.TestCase):
    def test_match_actions_pages(self):
        actions = [1, 0, 1, 0, 0]
        pages = [1, 2, 3]

        check = [(1, (0, 1)), (0, (1, 2)), (1, (2, 3)), (0, (3, 4)), (0, (4, 5))]
        result = NeighborActionSpace.match_actions_pages(actions, pages)
        # Algorithm:
        # size = 2, left = max(0, 1 - 2) = 0, right = 3 + 2 = 5
        # chunk_size = max(1, (5 - 0) // 5) = 1
        # range(left, right, chunk_size) = [ 0, 1, 2, 3, 4 ]
        # chunks = [ (0, 1), (1, 2), (2, 3), (3, 4), (4, 5) ]
        self.assertTrue(all([i == j for i, j in zip(check, result)]))


if __name__ == "__main__":
    unittest.main()
