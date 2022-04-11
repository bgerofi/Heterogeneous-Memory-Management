from gym.spaces import Space
import numpy as np
from intervaltree import IntervalTree, Interval
import unittest

"""
This module implements OpenAI gym action space and related actions.
"""


class DefaultActionSpace(Space):
    """
    This is the base action class.
    This class implements the action space and a default action (do nothing).
    The action space is simply an array of boolean values.
    This class is meant to be extended with a new `do_action()` method
    implementing a specific way to handle actions, i.e matching boolean
    values of actions into actions.
    """

    def __init__(
        self,
        num_actions,
        move_page_cost=10,
        page_size=1 << 14,
        seed=None,
    ):
        """
        Action space creation method.
        @arg num_actions: The size of the space in number of possible actions
        to perform.
        @arg move_page_cost: A constant penalty value for each page movement.
        @arg page_size: The base size of data to move.
        @arg seed: See `gym.spaces.Space.__init__()`
        """
        super().__init__([num_actions], np.uint8, seed)
        self._move_page_cost = move_page_cost
        self._page_size = page_size
        self._page_shift = int(np.round(np.log2(1 << 14)))
        self._page_mask = ~(page_size - 1)

    def do_action(self, action, memory, *args):
        """
        This is the method that implements the actions to perform.
        This method specializes the class to a specific range of actions.
        @arg action: A vector of boolean actions of this space size.
        @arg memory: A computer high bandwidth memory associated with the
        environment. This is used to perform allocation/free actions.
        See `memai.memory.Memory`.
        @arg args: This depends on the environment implementation.
        It currently is: `(observation, pages, count, t_ddr, t_hbm)`
        where:
        - observation is the last environment observation before this action
        is taken,
        - pages is a list of accessed page addresses associated with the
        observation,
        - count is the number of page accesses for each page,
        - t_ddr is the real execution time to perform the accesses in the
        ddr memory,
        - t_hbm is the real execution time to perform the accesses in the
        hbm memory.
        @return Number of pages moved times the cost of moving a page.
        """
        return 0

    def all_to_hbm_action(self):
        """
        Return the action vector of moving all the pages to hbm memory.
        """
        return np.ones(self.shape[0], dtype=np.uint8)

    def all_to_ddr_action(self):
        """
        Return the action vector of moving all the pages to ddr memory.
        """
        return np.zeros(self.shape[0], dtype=np.uint8)

    def sample(self) -> np.ndarray:
        """
        Return a random action vector.
        """
        return self.np_random.randint(0, 2, self._shape, dtype=self.dtype)

    def contains(self, x):
        if isinstance(x, np.ndarray):
            return x.shape == self.shape
        if isinstance(x, list):
            return len(x) == self.shape[0]
        return False


class NeighborActionSpace(DefaultActionSpace):
    """
    This class extends `DefaultActionSpace` class to implement `do_action()`
    method as follow.

    Compute the range "R" of accessed pages on last observation.
    Create a contiguous interval of chunks from (min_page_address - R) to
    (max_page_address - R) and of size "S" = (3*R / num_actions).
    Each action is therefore associated with a chunk of data of size "S" with
    a start and end address.
    This class with move data to ddr memory if action is 0 else to hbm memory.
    The pages that could be moved because target memory had space for it and
    did not already contain the page are counted toward the move_pages_penalty.
    """

    @staticmethod
    def match_actions_pages(actions, pages, addr_max=np.iinfo(np.uint64).max):
        """
        This function takes an action (vector of boolean) and a set of
        accessed pages and returns a list of tuples:
        `(action_boolean, (address_start, address_stop))` where
        `action_boolean` is 0 to move all pages from `address_start` to
        `address_stop` in the ddr memory and 1 to move them in the hbm memory.
        For each actions is associated a range of addresses. Ranges of
        addresses are contiguous, of the same size, and the total range of
        addresses spans three times the range "R" of input pages from the
        smallest input page address minus "R" to the highest input page
        address plus "R".
        """
        page_min = np.nanmin(pages)
        page_max = np.nanmax(pages)
        size = page_max - page_min

        page_min = 0 if page_min <= size else page_min - size
        page_max = addr_max if (addr_max - page_max) < size else page_max + size
        chunk_size = max(1, (page_max - page_min) // len(actions))

        return zip(
            actions.astype(bool),
            ((p, p + chunk_size) for p in range(page_min, page_max, chunk_size)),
        )

    def do_action(self, actions, memory, *args):
        """
        See class documentation.
        The function returns the penalty for successfully moved pages.
        """
        observation, pages, count, t_ddr, t_hbm = args
        if len(pages) == 0:
            return 0
        action_pages = list(NeighborActionSpace.match_actions_pages(actions, pages))

        rm_pages = [
            Interval(begin & self._page_mask, end & self._page_mask)
            for a, (begin, end) in action_pages
            if a == False
        ]
        add_pages = [
            Interval(begin & self._page_mask, end & self._page_mask)
            for a, (begin, end) in action_pages
            if a == True
        ]
        if len(rm_pages) > 0:
            free_size = memory.free(rm_pages)
            n_free_pages = free_size >> self._page_shift
        else:
            n_free_pages = 0
        if len(add_pages) > 0:
            alloc_size = memory.alloc(add_pages)
            n_alloc_pages = alloc_size >> self._page_shift
        else:
            n_alloc_pages = 0

        return self._move_page_cost * (n_free_pages + n_alloc_pages)


class TestNeighborActionSpace(unittest.TestCase):
    def test_match_actions_pages(self):
        actions = np.array([1, 0, 1, 0, 0], dtype=bool)
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
