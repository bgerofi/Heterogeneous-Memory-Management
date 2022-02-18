from memai import Estimator, TraceSet, Trace, IntervalDetector
from gym import Env, spaces


def define_actions(intervals, page_size):
    counter = 0
    actions_map = IntervalTree()

    for i in intervals:
        num_pages = int((i.end - i.begin) / page_size)
        actions_map.add(Interval(counter, counter + num_pages), i)
        counter += num_pages
    return counter, actions_map


class TraceEnv(Env):
    """ """

    def __init__(
        self,
        trace_ddr_file,
        trace_hbm_file,
        cpu_cycles_per_ms,
        window_length=50,
        compare_unit="ms",
        interval_distance=1 << 22,
        page_size=1 << 14,
    ):
        trace_ddr = Trace(trace_ddr_file, cpu_cycles_per_ms)
        trace_hbm = Trace(trace_hbm_file, cpu_cycles_per_ms)
        traces = TraceSet(trace_ddr, trace_hbm, window_length, compare_unit)

        self.traces = traces
        # self.intervals = (
        #     IntervalDetector(interval_distance, page_size)
        #     .append_addr(trace_ddr.virtual_addresses())
        #     .append_addr(trace_hbm.virtual_addresses())
        # )

        self._page_size_ = int(np.log2(page_size))
        self._hbm_intervals_ = IntervalTree()
        self._window_iterator_ = iter(traces)
        self._estimator_ = Estimator(self.traces, self._page_size_)
        self._move_pages_time_ = 0

    def reset(self, *args, **kwargs):
        self._hbm_intervals_ = IntervalTree()
        self._window_iterator_ = iter(traces)
        self._move_pages_time_ = 0
        try:
            return next(self._window_iterator_)
        except StopIteration:
            return None

    def step(self, action: ActType):
        try:
            window = next(self.traces)

            if window.is_empty():
                return self.step(action)

            # window.trace_ddr._data_["Offsets"] = list(
            #     self.intervals.offsetof(window.trace_ddr)
            # )
            # window.trace_hbm._data_["Offsets"] = list(
            #     self.intervals.offsetof(window.trace_hbm)
            # )

            penalty = 0
            self._move_pages_time_ += penalty
            reward = penalty  # Update with action to add penalty for movements.
            stop = False
            observation = window
        except StopIteration:
            reward = self._estimator_.estimate(self._hbm_intervals_)
            stop = True
            window = None
        debug_info = {}
        return window, reward, stop, debug_info

    def render(self, mode="human"):
        estimated_time = self._estimator_.estimate(self._hbm_intervals_)
        estimated_time -= self._move_pages_time_
        estimated_time /= 1000.0
        hbm_time = self.traces.timespan_hbm() / 1000.0
        ddr_time = self.traces.timespan_ddr() / 1000.0

        s = "Projected estimated time: {.2f}(s), "
        "DDR time: {.2f}(s), "
        "HBM time: {.2f}(s)".format(estimated_time, ddr_time, hbm_time)

        if mode == "ansi":
            return s
        elif mode == "human":
            print(s)
        else:
            raise ValueError("Rendering only supports 'human' or 'ansi'")
