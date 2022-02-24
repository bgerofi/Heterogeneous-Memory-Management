from memai import (
    Estimator,
    IntervalDetector,
    TraceSet,
    Trace,
    WindowObservationSpace,
    MovePagesActionSpace,
)
from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree


class TraceEnv(Env):
    def __init__(
        self,
        trace_set,
        page_size=1 << 14,
        interval_distance=1 << 22,
        observation_space=WindowObservationSpace(128, 128),
        action_space=MovePagesActionSpace(
            num_actions=128, move_page_cost=10, page_size=1 << 14
        ),
    ):
        # Gym specific attributes
        self.observation_space = observation_space
        self.action_space = action_space

        self._traces = trace_set
        self._interval_distance = interval_distance
        self._page_size = page_size
        self._reset()

    def _reset(self):
        self._mmap_intervals = IntervalDetector(
            self._interval_distance, self._page_size
        )
        self._hbm_intervals = IntervalTree()
        self._estimated_time = 0.0
        self._move_pages_time = 0.0
        self._previous_window_empty = False
        self._empty_window_start = 0
        self._windows = iter(self._traces)
        self._interval_observations = iter([])
        self._estimation = None
        self._interval = None

    def _next_window(self):
        window = next(self._windows)

        # Iterate empty windows.
        while window.is_empty():
            self._previous_window_empty = True
            window = next(self._windows)

        # Increment time spent on empty windows.
        if self._previous_window_empty == True:
            self._estimated_time += window.time_start() - self._empty_window_start
        self._previous_window_empty = False
        self._empty_window_start = window.time_end()

        # Update mmap_intervals
        self._mmap_intervals.append_addresses(window.trace_ddr.virtual_addresses())
        self._mmap_intervals.append_addresses(window.trace_hbm.virtual_addresses())

        # Compute time estimation for the window
        self._estimation = Estimator(window, int(np.log2(self._page_size))).estimate(
            self._hbm_intervals
        )
        self._estimated_time += self._estimation

        # For each mmapped interval build an observation
        observations = []
        for interval in self._mmap_intervals:
            addr = window.trace_ddr.virtual_addresses()
            addr = addr & self._mmap_intervals._page_mask
            addr = np.array([x if x in interval else -1 for x in addr], dtype=np.int64)
            addr = addr >> self._mmap_intervals._page_shift
            addr = addr - np.nanmin(addr)
            print(max(addr))
            print(interval.end - interval.begin)

            observation = self.observation_space.from_addresses(addr)
            observations.append(observation)
        self._interval_observations = zip(self._mmap_intervals, observations)

    def step(self, action):
        try:
            # Get next window, associated observations and estimator.
            interval, observation = next(self._interval_observations)

            # Do action (move pages) and retrieve move pages penalty.
            if self._interval is not None:
                move_pages_time = self.action_space.do_action(
                    action, self._interval, self._hbm_intervals
                )
            else:
                move_pages_time = 0
            self._interval = interval
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
                reward = -1.0 * self._estimated_time
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


if __name__ == "__main__":
    import argparse
    import time

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
        default=13,
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

    args = parser.parse_args()

    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms)
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
        args.window_len,
        args.compare_unit,
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

    hbm_time = hbm_trace.timespan()
    ddr_time = ddr_trace.timespan()
    simulated_time = env._estimated_time + env._move_pages_time

    print("Time DDR: {:.2f} (s)".format(ddr_time / 1000.0))
    print("Time HBM: {:.2f} (s)".format(hbm_time / 1000.0))
    print("Simulated Time: {:.2f} (s)".format(simulated_time / 1000.0))
    print("\tEstimated Time: {:.2f} (s)".format(env._estimated_time / 1000.0))
    print("\tMove Pages Time: {:.2f} (s)".format(env._move_pages_time / 1000.0))
    print("Simulation Time: {:.2f} (s)".format(simulation_time))
