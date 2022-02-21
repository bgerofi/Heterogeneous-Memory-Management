from memai import Estimator, TraceSet, Trace, IntervalDetector
from gym import Env, spaces


class EstimatorIter:
    def __new__(self, traces, page_size, max_window_len, access_intervals):
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
                observation = TraceEnv._observation_(
                    window, max_window_len, access_intervals
                )
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
        managed_intervals=None,
        interval_distance=1 << 22,
        page_size=1 << 14,
        move_page_penalty=10.0,
    ):
        trace_ddr = Trace(trace_ddr_file, cpu_cycles_per_ms)
        trace_hbm = Trace(trace_hbm_file, cpu_cycles_per_ms)
        self.traces = TraceSet(
            trace_ddr, trace_hbm, window_length, compare_unit="accesses"
        )

        # The intervals where data is mapped.
        if managed_intervals is not None:
            self._access_intervals_ = IntervalDetector(
                managed_intervals, page_size, interval_distance
            )
        else:
            self._access_intervals_ = (
                IntervalDetector(interval_distance, page_size)
                .append_addr(trace_ddr.virtual_addresses())
                .append_addr(trace_hbm.virtual_addresses())
            )
        self._hbm_intervals_ = IntervalTree()
        self._window_len_ = window_len
        self._estimator_iterator_ = EstimatorIter(
            self.traces, page_size, self._window_len, self._access_intervals_
        )
        self._move_page_penalty_ = float(move_page_penalty)
        self._move_pages_time_ = 0.0
        self._estimated_time_ = self._estimator_iterator_._empty_time_

        # Gym specific attributes
        pages_per_interval = self._access_intervals_.max_interval_len()
        self.observation_space = spaces.MultiDiscrete(
            np.repeat(pages_per_interval, window_length)
        )
        self.action_space = spaces.MultiBinary(
            np.repeat(num_intervals, pages_per_interval)
        )

    @staticmethod
    def _observation_(window, max_window_len, access_intervals):
        """
        Translate a trace window into an observation usable by the AI.
        """
        page_offset_ids = [
            access_intervals.interval_addr_index(addr)
            for addr in window.trace_ddr.virtual_addressses()
        ]
        if len(page_offset_ids) < max_window_len:
            page_offset_ids += np.repeat(-1, max_window_len - len(page_offset_ids))
        elif len(page_offset_ids) > max_window_len:
            raise ValueError("The trace window length exceeds the set window length.")
        return page_offset_ids

    def _move_pages_(self, action):
        """
        Move pages in self._hbm_intervals_ according to `action` and
        return the time spent moving pages in milliseconds.
        """

        num_moves = 0
        for actions, interval in zip(action, self._access_intervals_):
            num_pages = self._access_intervals_.interval_len(interval)
            low = self._access_intervals_.interval_addr_at(range(num_pages), interval)
            high = low + self._access_intervals_._page_size_

            for do_set, l, h in zip(actions, low, high):
                page = Interval(l, h)
                is_set = page in self._hbm_intervals_
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
            move_pages_time = self._move_pages_(action)
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
