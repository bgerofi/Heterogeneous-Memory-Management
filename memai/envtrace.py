import numpy as np
import pandas as pd
import tqdm

from memai import (
    Estimator,
    IntervalDetector,
    WindowIterator,
    WindowObservationSpace,
)


class EnvTrace:
    OBSERVATION = "Observation"
    PAGE_ACCESSES = "PageAccesses"
    TIME_DDR = "TimeDDR"
    TIME_HBM = "TimeHBM"

    def __init__(
        self,
        observation_space=WindowObservationSpace(128, 128),
        page_size=1 << 14,
        interval_distance=1 << 22,
    ):
        self.observation_space = observation_space
        self.observations = []
        self.pages_accesses = []
        self.t_ddr = []
        self.t_hbm = []
        self.intervals = IntervalDetector(interval_distance, page_size)

    def _append_(self, observation, pages_accesses, t_ddr, t_hbm):
        self.observations.append(observation)
        self.pages_accesses.append(pages_accesses)
        self.t_ddr.append(t_ddr)
        self.t_hbm.append(t_hbm)

    def __eq__(self, other):
        if not all(self.t_hbm == other.t_hbm):
            return False
        if not all(self.t_ddr == other.t_ddr):
            return False
        if not all(self.observations == other.observations):
            return False
        if not all(self.pages_accesses == other.pages_accesses):
            return False
        return True

    def append_window(self, window):
        if window.is_empty():
            self._append_(
                self.observation_space.empty(), [], window.t_ddr, window.t_hbm
            )
            return

        window_pages = window.addresses & self.intervals.page_mask
        self.intervals.append_addresses(window_pages)

        for interval in self.intervals:
            interval_pages = []
            interval_addr = []
            min_addr = np.inf

            for p in window_pages:
                if p in interval:
                    interval_pages.append(p)
                    addr = p >> intervals.page_shift
                    interval_addr.append(addr)
                    min_addr = min(min_addr, addr)
                else:
                    interval_addr.append(-1)

            if len(interval_pages) == 0:
                continue

            interval_addr = np.array(interval_addr) - min_addr
            page, count = np.unique(np.array(interval_pages), return_counts=True)
            self._append_(
                self.observation_space.from_addresses(interval_addr),
                list(zip(page, count)),
                window.t_ddr,
                window.t_hbm,
            )

    def as_dict(self):
        return {
            EnvTrace.OBSERVATION: self.observations,
            EnvTrace.PAGE_ACCESSES: self.pages_accesses,
            EnvTrace.TIME_DDR: self.t_ddr,
            EnvTrace.TIME_HBM: self.t_hbm,
        }

    def as_pandas(self):
        return pd.DataFrame(self.as_dict())

    @staticmethod
    def from_trace_set(
        trace_set,
        page_size=1 << 14,
        interval_distance=1 << 22,
        compare_unit="ms",
        window_len=None,
        observation_space=WindowObservationSpace(128, 128),
    ):
        env_t = EnvTrace(observation_space)
        windows = WindowIterator(trace_set, compare_unit, window_len)
        progress_bar = tqdm.tqdm()
        last_timestamp = int(windows._bounds_fn_(trace_set.trace_ddr)[1])
        progress_bar.reset(total=last_timestamp)

        for window in windows:
            env_t.append_window(window)
            progress_bar.update(int(window.t_ddr))

        progress_bar.update(last_timestamp)
        progress_bar.clear()
        return env_t

    @staticmethod
    def from_pandas(filename, page_size=1 << 14, interval_distance=1 << 22):
        df = pd.read_feather(filename)
        colnames = [
            EnvTrace.OBSERVATION,
            EnvTrace.PAGE_ACCESSES,
            EnvTrace.TIME_DDR,
            EnvTrace.TIME_HBM,
        ]
        if not all(df.columns == colnames):
            raise ValueError(
                "Invalid dataframe columns name. Expected one of: {}".format(colnames)
            )
        if len(df) == 0:
            raise ValueError("Empty traces not supported.")

        first_obs = df[EnvTrace.OBSERVATION].iloc[0]
        nrow, ncol = first_obs.shape
        env_t = EnvTrace(
            WindowObservationSpace(nrow, ncol), page_size, interval_distance
        )
        env_t.observations = df[EnvTrace.OBSERVATION].values
        env_t.pages_accesses = df[EnvTrace.PAGE_ACCESSES].values
        env_t.t_ddr = df[EnvTrace.TIME_DDR].values
        env_t.t_hbm = df[EnvTrace.TIME_HBM].values
        env_t.intervals.append_addresses(
            [x[0] for p in env_t.pages_accesses for x in p]
        )
        return env_t
