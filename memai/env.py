from memai import (
    Estimator,
    IntervalDetector,
    TraceSet,
    WindowIterator,
    Trace,
    WindowObservationSpace,
    MovePagesActionSpace,
)
from gym import Env, spaces
import numpy as np
import tqdm
from intervaltree import IntervalTree, Interval


class TraceEnv(Env):
    def __init__(
        self,
        trace_set,
        page_size=1 << 14,
        interval_distance=1 << 22,
        window_len=1000,
        compare_unit="ms",
        observation_space=WindowObservationSpace(128, 128),
        action_space=MovePagesActionSpace(
            num_actions=128, move_page_cost=10, page_size=1 << 14
        ),
    ):
        # Gym specific attributes
        self.observation_space = observation_space
        self.action_space = action_space

        self.window_len = window_len
        self.compare_unit = compare_unit
        self._traces = trace_set
        self._progress_bar = tqdm.tqdm()
        self._reset(page_size, interval_distance)

    def _reset(self, page_size, interval_distance):
        self._mmap_intervals = IntervalDetector(interval_distance, page_size)
        self._hbm_intervals = IntervalTree()
        self._estimated_time = 0.0
        self._move_pages_time = 0.0

        self._previous_window_empty = False
        self._empty_window_start = 0
        self._progress_bar.reset(total=int(self._traces.time_end()))
        self._windows = iter(
            WindowIterator(self._traces, self.compare_unit, self.window_len)
        )
        self._interval_observations = iter([])
        self.previous_interval = None

    @property
    def page_size(self):
        return self._mmap_intervals.page_size

    @property
    def page_shift(self):
        return self._mmap_intervals.page_shift

    @property
    def page_mask(self):
        return self._mmap_intervals.page_mask

    @property
    def interval_distance(self):
        return self._mmap_intervals.interval_distance

    @property
    def estimated_time(self):
        return self._estimated_time

    @property
    def move_pages_time(self):
        return self._move_pages_time

    def next_observation(self):
        try:
            interval, observation = next(self._interval_observations)
            self.previous_interval = interval
            return observation
        except StopIteration:
            self.next_window()
            return self.next_observation()

    def next_window(self):
        while True:
            window = next(self._windows)
            if window.is_empty():
                # Increment time spent on empty windows.
                self._estimated_time += window.t_ddr
            else:
                break

        # Update mmap_intervals
        self._mmap_intervals.append_addresses(window.addresses)

        # Compute time estimation for the window
        self._estimated_time += Estimator.estimate_window(
            window, self._mmap_intervals.page_mask, self._hbm_intervals
        )

        # For each mmapped interval build an observation
        observations = []
        for interval in self._mmap_intervals:
            addr = window.addresses >> self.page_shift
            _interval = Interval(
                interval.begin >> self.page_shift, interval.end >> self.page_shift
            )
            addr = np.array([x if x in _interval else -1 for x in addr], dtype=np.int64)
            try:
                min_addr = np.nanmin([x for x in addr if x >= 0])
                interval = Interval(
                    min_addr << self.page_shift, np.nanmax(addr) << self.page_shift
                )
                addr = addr - min_addr
                observation = self.observation_space.from_addresses(addr)
                observations.append((interval, observation))
            except ValueError:
                # If no address is greater there is no address in this interval.
                continue

        self._interval_observations = iter(observations)
        self._progress_bar.update(int(window.t_ddr))

    def step(self, action):
        try:
            # Do action (move pages) and retrieve move pages penalty.
            if self.previous_interval is not None:
                move_pages_time = self.action_space.do_action(
                    action, self.previous_interval, self._hbm_intervals
                )
            else:
                move_pages_time = 0
            self._move_pages_time += move_pages_time

            # Get next window, associated observations and estimator.
            observation = self.next_observation()

            # The reward is negative for moving pages (we want to avoid that)
            reward = -1.0 * move_pages_time
            stop = False

        except StopIteration:
            self._progress_bar.update(self._progress_bar.total - self._progress_bar.n)
            self._progress_bar.clear()
            # The reward is negative for the total elapsed time
            # (we want to minimize elapsed time).
            reward = -1.0 * self.estimated_time
            stop = True
            observation = None

        debug_info = {}
        return observation, reward, stop, debug_info

    def reset(
        self,
        *,
        seed=None,
        return_info=False,
        options=None,
    ):
        super().reset(seed=seed)
        self._reset(self.page_size(), self.interval_distance())
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


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddr-input",
        metavar="<file>",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in ddr memory."
        ),
    )
    parser.add_argument(
        "--hbm-input",
        metavar="<file>",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in hbm memory."
        ),
    )
    parser.add_argument(
        "--page-size",
        metavar="<int>",
        default=1 << 14,
        type=int,
        help="The size of a page in number of bits.",
    )
    parser.add_argument(
        "--cpu-cycles-per-ms",
        metavar="<int>",
        default=1400000,
        type=int,
        help="CPU cycles per millisecond (default: 1,400,000 for KNL).",
    )
    parser.add_argument(
        "--window-len",
        metavar="<int>",
        default=None,
        type=int,
        help="Comparison window length.",
    )
    parser.add_argument(
        "--compare-unit",
        default="ms",
        choices=["accesses", "ms", "instrs"],
        type=str,
        help="Comparison length unit.",
    )
    parser.add_argument(
        "--mmap-distance",
        metavar="<int>",
        default=1 << 22,
        type=int,
        help="The minimum size between two disjoint mappings in the address space.",
    )
    parser.add_argument(
        "--observation-columns",
        metavar="<int>",
        default=128,
        type=int,
        help="The width of an observation matrix. Windows are scaled in the compare_unit dimension to fit the width.",
    )
    parser.add_argument(
        "--observation-rows",
        metavar="<int>",
        default=128,
        type=int,
        help="The height of an observation matrix. mmap_intervals are scaled to to fit the height.",
    )
    parser.add_argument(
        "--num-actions",
        metavar="<int>",
        default=128,
        type=int,
        help="The number of possible actions.",
    )
    parser.add_argument(
        "--move-page-cost",
        metavar="<int>",
        default=10,
        type=int,
        help="The cost of moving a page in milliseconds.",
    )
    parser.add_argument(
        "--tracemalloc",
        default=False,
        action="store_true",
        help="Trace and print peak memory usage. This slows down execution.",
    )

    args = parser.parse_args()

    if args.tracemalloc:
        tracemalloc.start()

    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms)
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
    )

    observation_space = WindowObservationSpace(
        args.observation_rows, args.observation_columns
    )

    action_space = MovePagesActionSpace(
        args.num_actions, args.move_page_cost, args.page_size
    )
    env = TraceEnv(
        traces,
        args.page_size,
        args.mmap_distance,
        observation_space=observation_space,
        action_space=action_space,
    )

    stop = False
    t = time.time()
    while not stop:
        observation, reward, stop, debug_info = env.step(action_space.sample())
    simulation_time = time.time() - t

    if args.tracemalloc:
        size, peak_size = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    hbm_time = hbm_trace.timespan()
    ddr_time = ddr_trace.timespan()
    simulated_time = env._estimated_time + env._move_pages_time

    print("Time DDR: {:.2f} (s)".format(ddr_time / 1000.0))
    print("Time HBM: {:.2f} (s)".format(hbm_time / 1000.0))
    print("Simulated Time: {:.2f} (s)".format(simulated_time / 1000.0))
    print("\tEstimated Time: {:.2f} (s)".format(env._estimated_time / 1000.0))
    print("\tMove Pages Time: {:.2f} (s)".format(env._move_pages_time / 1000.0))
    print("Simulation Time: {:.2f} (s)".format(simulation_time))
    if args.tracemalloc:
        print("Peak Memory Usage: {:.2f} (MB)".format(peak_size / 1e6))
