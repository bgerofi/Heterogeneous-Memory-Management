from memai import Estimator, IntervalDetector, TraceSet, Trace, WindowObservationSpace
from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree


class TraceEnv(Env):
    def __init__(
        self,
        trace_ddr_file,
        trace_hbm_file,
        cpu_cycles_per_ms,
        window_length=300,
        compare_unit="ms",
        page_size=1 << 14,
        move_page_penalty=10.0,
        interval_distance=1 << 22,
        observation_space=WindowObservationSpace(128, 128),
        action_space=ActionSpace(128),
    ):
        # Gym specific attributes
        self.observation_space = observation_space
        self.action_space = None  # TODO

        trace_ddr = Trace(trace_ddr_file, cpu_cycles_per_ms)
        trace_hbm = Trace(trace_hbm_file, cpu_cycles_per_ms)

        self._traces = TraceSet(trace_ddr, trace_hbm, window_length, compare_unit)
        self._interval_distance = interval_distance
        self._page_size = page_size
        self._move_page_penalty = float(move_page_penalty)
        self._reset()

    def _reset(self):
        self._mmap_intervals = IntervalDetector(
            self._interval_distance, self._page_size
        )
        self._hbm_intervals = IntervalTree()
        self._estimated_time = 0.0
        self._move_pages_time = 0.0
        self._empty_time = 0.0
        self._previous_window_empty = False
        self._empty_window_start = 0
        self._windows = iter(self._traces)
        self._interval_observations = iter([])
        self._estimation = None

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

        # Compute time estimation for the window
        self._estimation = Estimator(window, int(np.log2(self._page_size))).estimate(self._hbm_intervals)
        self._estimated_time_ += self._estimation


        # For each mmapped interval build an observation
        observations = []
        for interval in self._mmap_intervals:
            addr = window.virtual_addresses().copy()
            apply_along_axis(lambda x: x if x in interval else np.nan, 0, addr)
            observation = self.observation_space.from_addresses(addr)
            observations.append(observation)
        self._interval_observations = zip(self._mmap_intervals, observations)

    def step(self, action: ActType):
        try:
            # Get next window, associated observations and estimator.
            interval, observation = next(self._interval_observations)

            # Do action (move pages) and retrieve move pages penalty.
            move_pages_time = self._move_pages(action, interval)
            self._move_pages_time_ += move_pages_time

            # The reward is negative for moving pages (we want to avoid that)
            reward = -1.0 * move_pages_time
            stop = False
        except StopIteration:
            try:
                self._next_window()
                return step(action)
            except StopIteration:
                # The reward is negative for the total elapsed time
                # (we want to minimize elapsed time).
                reward = -1.0 * self._estimated_time_
                stop = True
                observation = None

        debug_info = {}
        return observation, reward, stop, debug_info

    def _move_pages(self, action, interval):
        """
        Move pages in self._hbm_intervals according to `action` and
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
