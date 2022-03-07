import pandas as pd
import tqdm
from intervaltree import IntervalTree, Interval

from memai import (
    Estimator,
    IntervalDetector,
    WindowIterator,
    WindowObservationSpace,
)


class EnvTrace:
    @staticmethod
    def as_dict(observations, page_accesses, t_ddr, t_hbm):
        return {
            "Observation": observations,
            "PageAccesses": page_accesses,
            "TimeDDR": t_ddr,
            "TimeHBM": t_hbm,
        }

    @staticmethod
    def as_pandas(observations, page_accesses, t_ddr, t_hbm):
        return pd.DataFrame(
            EnvTraceSample.as_dict(observations, page_accesses, t_ddr, t_hbm)
        )

    @staticmethod
    def from_trace_set(
        trace_set,
        page_size=1 << 14,
        interval_distance=1 << 22,
        compare_unit="ms",
        window_len=None,
        observation_space=WindowObservationSpace(128, 128),
    ):
        intervals = IntervalDetector(interval_distance, page_size)
        windows = WindowIterator(trace_set, compare_unit, window_len)
        observations = []
        page_accesses = []
        t_ddr = []
        t_hbm = []

        progress_bar = tqdm.tqdm()
        progress_bar.reset(total=int(WindowIterator._bounds_fn_(trace_set.trace_ddr)))

        for window in windows:
            if window.is_empty():
                observations.append(observation_space.empty())
                page_accesses.append([])
                t_ddr.append(window.t_ddr)
                t_hbm.append(window.t_hbm)
                continue

            window_pages = window.addresses & intervals.page_mask
            intervals.append_addresses(window_pages)

            for interval in intervals:
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
                observations.append(observation_space.from_addresses(interval_addr))
                page_accesses.append(list(zip(page, count)))
                t_ddr.append(window.t_ddr)
                t_hbm.append(window.t_hbm)
            progress_bar.update(int(window.t_ddr))

        progress_bar.update(progress_bar.total - progress_bar.n)
        progress_bar.clear()

        return EnvTrace.as_pandas(observations, page_accesses, t_ddr, t_hbm)
