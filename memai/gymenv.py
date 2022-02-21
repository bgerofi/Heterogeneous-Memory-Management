from memai import Estimator, TraceSet, Trace, IntervalDetector, AddressSpace
from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree

class AddressSpace:
    def __init__(self, intervals, page_size=1 << 14):
        self._index_addr_ = []
        self._addr_index_ = {}
        self._page_mask_ = ~(page_size - 1)
        self.page_size = page_size

        i = 0
        for interval in intervals:
            low = interval.begin & self._page_mask_
            high = interval.end & self._page_mask_
            high = high + page_size if high != interval.end else high
            self.index_addr += list(range(low, high, page_size))
            self._addr_index_.update(
                {addr: i + j for j, addr in enumerate(range(low, high, page_size))}
            )
            i += (high - low) / page_size

    def addr_index(self, addr):
        try:
            return self._addr_index_[addr & self._page_mask_]
        except KeyError:
            raise ValueError("Invalid address not in address space.")

    def index_addr(self, i):
        return self._index_addr_[i]

    def __len__(self):
        return len(self._index_address_)

    def __getitem__(self, addr):
        return self.addr_index(addr)


class IntervalDetector:
    def __init__(self, interval_distance=1 << 22, page_size=1 << 14):
        self.intervals = IntervalTree()
        self._page_mask_ = ~(page_size - 1)
        self._interval_distance_ = interval_distance * page_size

    def append_addresses(self, vaddr):
        low_vaddr, high_vaddr = self._page_bounds_(vaddr)
        for low, high in zip(low_vaddr, high_vaddr):
            self.intervals[low:high] = 0
        self.intervals.merge_overlaps()
        return self

    def _page_bounds_(self, vaddr):
        if isinstance(vaddr, list):
            vaddr = np.array(vaddr)
        if isinstance(vaddr, (np.ndarray, np.array)):
            vaddr = vaddr.reshape(1, len(b))
        low = (vaddr & self._page_mask_) - self._interval_distance_
        low = np.apply_along_axis(lambda i: max(i, 0), 0, low)
        high = (vaddr + self._interval_distance_) & self._page_mask_
        return low, high


class EstimatorIter:
    def __new__(self, traces, page_size, trace_env):
        self._empty_time_ = 0.0
        self._estimators_ = []
        self._observations_ = []
        self._index_ = 0

        for i, item in enumerate(EstimatorWindowsIter(application_traces, page_size)):
            window, empty_time, fast_time, ddr_time, hbm_time, pages = item

            self._empty_time_ += empty_time
            if len(pages) != 0:
                estimator = Estimator.from_iter(
                    0.0, fast_time, ddr_time, hbm_time, pages
                )
                observation = trace_env.observation(window)
                self._estimators_.append(estimator)
                self._observations_.append(observation)

    def __iter__(self):
        try:
            observation = self._observations_[self._index_]
            estimator = self._estimators_[self._index_]
            ret = (observation, estimator)
            self._index_ += 1
            return ret
        except IndexError:
            raise StopIteration

    def reset(self):
        self._index_ = 0


class TraceEnv(Env):
    def __init__(
        self,
        trace_ddr_file,
        trace_hbm_file,
        cpu_cycles_per_ms,
        window_length=50,
        max_managed_address=None,
        managed_intervals=None,
        interval_distance=1 << 22,
        page_size=1 << 14,
        move_page_penalty=10.0,
    ):
        # The intervals where data is mapped.
        if managed_intervals is not None:
            self._address_space_ = AddressSpace(managed_intervals, page_size)
        else:
            intervals = (
                IntervalDetector(interval_distance, page_size)
                .append_addr(trace_ddr.virtual_addresses())
                .append_addr(trace_hbm.virtual_addresses())
                .intervals
            )
            self._address_space_ = AddressSpace(intervals, page_size)
        num_pages = len(self._address_space_)
        if max_managed_address is None:
            max_managed_address = num_pages
        if num_pages > max_managed_address:
            raise ValueError(
                "The maximum amount of address that can be managed exceeeds the amount of pages to manage."
            )
        self._num_pages_ = max_managed_address

        trace_ddr = Trace(trace_ddr_file, cpu_cycles_per_ms)
        trace_hbm = Trace(trace_hbm_file, cpu_cycles_per_ms)
        self.traces = TraceSet(
            trace_ddr, trace_hbm, window_length, compare_unit="accesses"
        )

        self._observations_shape_ = (max_managed_address, window_len)
        self._hbm_intervals_ = IntervalTree()
        self._move_page_penalty_ = float(move_page_penalty)
        self._move_pages_time_ = 0.0
        self._estimator_iterator_ = EstimatorIter(
            self.traces, page_size, observation_shape, self._address_space_
        )
        self._estimated_time_ = self._estimator_iterator_._empty_time_

        # Gym specific attributes
        self.observation_space = spaces.MultiBinary(list(self._observations_shape_))
        self.action_space = spaces.MultiBinary([max_managed_address])

    def observation(self, window):
        """
        Translate a trace window into an observation usable by the AI.
        """
        ret = np.zeros(self._observations_shape_, dtype=self.observation_space.dtype)
        for i, addr in enumerate(window.trace_ddr.virtual_addressses()):
            ret[self._address_space_.addr_index(addr), i] = 1
        return ret

    def move_pages(self, action):
        """
        Move pages in self._hbm_intervals_ according to `action` and
        return the time spent moving pages in milliseconds.
        """

        num_moves = 0
        for do_set, addr in zip(action, self._address_space_._index_addr_):
            is_set = page in self._hbm_intervals_
            page = Interval(addr, addr + address_space.page_size)
            if do_set:
                if not is_set:
                    num_moves += 1
                    self._hbm_intervals_.add(page)
            else:
                if is_set:
                    num_moves += 1
                    self._hbm_intervals_.remove(page)

        self._hbm_intervals_.merge_overlaps()
        return num_moves * self._move_page_penalty_

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._hbm_intervals_ = IntervalTree()
        self._estimator_iterator_.reset()
        self._move_pages_time_ = 0.0
        self._estimated_time_ = self._estimator_iterator_._empty_time_
        try:
            # The void action when the hbm is empty.
            action = np.zeros(self.action_space.n, dtype=self.action_space.dtype)
            obs, _, _, _ = self.step(action)
            return obs
        except StopIteration:
            return None

    def step(self, action: ActType):
        try:
            # Get next window.
            observation, estimator = next(self._estimator_iterator_)
            self._estimated_time_ += empty_time
            if estimator is None:
                return self.step(action)

            # Estimate time elapsed before action.
            estimated_time = estimator.estimate(self._hbm_intervals)
            self._estimated_time_ += estimated_time

            # Do action (move pages) and retrieve move pages penalty.
            move_pages_time = self.move_pages(action)
            self._move_pages_time_ += move_pages_time

            # The reward is negative for moving pages (we want to avoid that)
            # And it is also negative for the elapsed time
            # (we want to minimize elapsed time).
            reward = -1.0 * move_pages_time - estimated_time
            stop = False
        except StopIteration:
            reward = 0
            stop = True
            observation = None
        debug_info = {}
        return observation, reward, stop, debug_info

    def render(self, mode="human"):
        total_time = self._estimated_time_ / 1000.0
        penalty_time = self._move_pages_time_ / 1000.0

        s = "Estimated time = {.2f}(s): ({.2f}(s) + {.2f}(s) penalty on page moves)".format(
            total_time + penalty_time, total_time, penalty_time
        )

        if mode == "ansi":
            return s
        elif mode == "human":
            print(s)
        else:
            raise ValueError("Rendering only supports 'human' or 'ansi'")
