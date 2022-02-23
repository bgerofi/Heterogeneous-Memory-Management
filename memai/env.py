from memai import Estimator, TraceSet, Trace, WindowObservationSpace
from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree


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


class TraceEnv(Env):
    def __init__(
        self,
        trace_ddr_file,
        trace_hbm_file,
        cpu_cycles_per_ms,
        window_length=300,
        compare_unit="ms",
        observation_space=WindowObservationSpace(128, 128),
        page_size=1 << 14,
        move_page_penalty=10.0,
        interval_distance=1 << 22,
    ):
        # Gym specific attributes
        self.observation_space = WindowObservationSpace(
            observation_rows, observation_cols
        )
        self.action_space = None  # TODO

        trace_ddr = Trace(trace_ddr_file, cpu_cycles_per_ms)
        trace_hbm = Trace(trace_hbm_file, cpu_cycles_per_ms)

        self._traces = TraceSet(trace_ddr, trace_hbm, window_length, compare_unit)
        self._interval_distance = interval_distance
        self._page_size = page_size
        self._move_page_penalty = float(move_page_penalty)
        self._empty_time = 0.0
        self._reset()

    def _reset(self):
        self._mmap_intervals = IntervalDetector(
            self._interval_distance, self._page_size
        )
        self._hbm_intervals = IntervalTree()
        self._index = 0
        self._estimated_time = 0.0
        self._move_pages_time = 0.0
        self._previous_window_empty = False
        self._empty_window_start = -1
        self._windows = iter(self._traces)
        self._estimator = None
        self._observations = next(iter([]))

    def _next_window(self):
        window = next(self._windows)

        # Iterate empty windows.
        while window.is_empty():
            self._previous_window_empty = True
            window = next(self._windows)

        # Increment time spent on empty windows.
        if self._previous_window_empty == True:
            self._empty_time += window.time_start() - self._empty_window_start
        self._previous_window_empty = False
        self._empty_window_start = window.time_end()

        # Update mmap_intervals
        self._mmap_intervals.append_addresses(
            window.trace_ddr.virtual_addresses()
        ).append_addresses(window.trace_hbm.virtual_addresses())

        # Build an estimator of the window
        self._estimator = Estimator(window, int(np.log2(self._page_size_)))

        # For each interval build an observation
        observations = []
        for interval in self._mmap_intervals:
            addr = window.virtual_addresses().copy()
            apply_along_axis(lambda x: x if x in interval else np.nan, 0, addr)
            observation = self.observation_space.from_addresses(addr)
            observations.append(observation)
        self._observations = iter(observations)

    def step(self, action: ActType):
        try:
            # Get next window, associated observations and estimator.
            observation = next(self._observations)

            # Estimate time elapsed before action.
            estimated_time = self._estimator.estimate(self._hbm_intervals)

            # Do action (move pages) and retrieve move pages penalty.
            move_pages_time = self._move_pages(action)
            self._move_pages_time_ += move_pages_time
            self._estimated_time_ += estimated_time

            # The reward is negative for moving pages (we want to avoid that)
            # And it is also negative for the elapsed time
            # (we want to minimize elapsed time).
            reward = -1.0 * (move_pages_time + estimated_time)
            stop = False
        except StopIteration:
            try:
                self._next_window()
                return step(action)
            except StopIteration:
                reward = 0
                stop = True
                observation = None

        debug_info = {}
        return observation, reward, stop, debug_info

    def _move_pages(self, action):
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
        self._reset()
        try:
            # The void action when the hbm is empty.
            action = np.zeros(self.action_space.n, dtype=self.action_space.dtype)
            obs, _, _, _ = self.step(action)
            return obs
        except StopIteration:
            return None

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
