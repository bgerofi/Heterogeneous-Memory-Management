import numpy as np
import pandas as pd
import tqdm

from memai import (
    Estimator,
    IntervalDetector,
    WindowIterator,
    WindowObservationSpace,
)


class Preprocessing:
    OBSERVATION = "Observation"
    PAGE_ACCESSES = "PageAccesses"
    TIME_DDR = "TimeDDR"
    TIME_HBM = "TimeHBM"

    def __init__(
        self,
        observation_space=WindowObservationSpace(128, 128),
        page_size=1 << 20,
        interval_distance=1 << 28,
    ):
        self.observation_space = observation_space
        self.observations = []
        self.pages_accesses = []
        self.t_ddr = []
        self.t_hbm = []
        self.intervals = IntervalDetector(interval_distance, page_size)

    def __iter__(self):
        return zip(self.observations, self.pages_accesses, self.t_ddr, self.t_hbm)

    def _append_(self, observation, pages_accesses, t_ddr, t_hbm):
        self.observations.append(observation)
        self.pages_accesses.append(pages_accesses)
        self.t_ddr.append(t_ddr)
        self.t_hbm.append(t_hbm)

    def __eq__(self, other):
        if not all(np.array(self.t_hbm) == np.array(other.t_hbm)):
            return False
        if not all(np.array(self.t_ddr) == np.array(other.t_ddr)):
            return False
        observations = np.array([o.flatten() for o in self.observations]).flatten()
        other_observations = np.array(
            [o.flatten() for o in other.observations]
        ).flatten()
        if not all(observations == other_observations):
            return False

        accesses = np.concatenate([np.array(p).flatten() for p in self.pages_accesses])
        other_accesses = np.concatenate(
            [np.array(p).flatten() for p in other.pages_accesses]
        )
        if not all(accesses == other_accesses):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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
                if interval.contains_point(p):
                    interval_pages.append(p)
                    addr = p >> self.intervals.page_shift
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
            Preprocessing.OBSERVATION: self.observations,
            Preprocessing.PAGE_ACCESSES: self.pages_accesses,
            Preprocessing.TIME_DDR: self.t_ddr,
            Preprocessing.TIME_HBM: self.t_hbm,
        }

    def as_pandas(self, output_file=None):
        df = pd.DataFrame(self.as_dict())
        if output_file is not None:
            df.to_feather(output_file)
        return df

    @staticmethod
    def from_trace_set(
        trace_set,
        page_size=1 << 20,
        interval_distance=1 << 28,
        compare_unit="ms",
        window_len=None,
        observation_space=WindowObservationSpace(128, 128),
    ):
        env_t = Preprocessing(observation_space)
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
    def from_pandas(filename, page_size=1 << 20, interval_distance=1 << 28):
        if isinstance(filename, pd.DataFrame):
            df = filename
        elif isinstance(filename, str):
            df = pd.read_feather(filename)
        else:
            raise ValueError("Expected either a pandas dataframe or a filename.")
        colnames = [
            Preprocessing.OBSERVATION,
            Preprocessing.PAGE_ACCESSES,
            Preprocessing.TIME_DDR,
            Preprocessing.TIME_HBM,
        ]
        if not all(df.columns == colnames):
            raise ValueError(
                "Invalid dataframe columns name. Expected one of: {}".format(colnames)
            )
        if len(df) == 0:
            raise ValueError("Empty traces not supported.")

        first_obs = df[Preprocessing.OBSERVATION].iloc[0]
        nrow, ncol = first_obs.shape
        env_t = Preprocessing(
            WindowObservationSpace(nrow, ncol), page_size, interval_distance
        )
        env_t.observations = df[Preprocessing.OBSERVATION].values
        env_t.pages_accesses = df[Preprocessing.PAGE_ACCESSES].values
        env_t.t_ddr = df[Preprocessing.TIME_DDR].values
        env_t.t_hbm = df[Preprocessing.TIME_HBM].values
        env_t.intervals.append_addresses(
            np.unique(
                np.array([x[0] for p in env_t.pages_accesses for x in p]).flatten()
            )
        )
        return env_t


if __name__ == "__main__":
    import argparse
    from memai import Trace, TraceSet
    from memai.options import *
    import sys

    parser = argparse.ArgumentParser()
    add_traces_input_args(parser)
    add_window_args(parser)
    parser.add_argument(
        "--output",
        metavar="<file>",
        type=str,
        default=None,
        help=("Where to write the obtain pandas dataframe."),
    )
    parser.add_argument(
        "--interval-distance",
        metavar="<int>",
        type=int,
        default=20,
        help=(
            "Minimum distance seperating two non contiguous chunks of memory."
            "The value is the exponent of a power of two."
        ),
    )
    parser.add_argument(
        "--observation-rows",
        metavar="<int>",
        type=int,
        default=128,
        help=("The number of row in an observation (window of timestamp X address)."),
    )
    parser.add_argument(
        "--observation-columns",
        metavar="<int>",
        type=int,
        default=128,
        help=(
            "The number of columns in an observation (window of timestamp X address)."
        ),
    )

    args = parser.parse_args()

    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms)
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
    )
    if args.interval_distance <= args.page_size:
        raise ValueError("Interval distance must be greater than page size.")

    print("Processing traces.")
    env_t = Preprocessing.from_trace_set(
        traces,
        args.page_size,
        args.interval_distance,
        args.compare_unit,
        args.window_len,
        observation_space=WindowObservationSpace(
            args.observation_rows, args.observation_columns
        ),
    )

    if args.output is not None:
        print("Export processed traces to: {}".format(args.output))
    df = env_t.as_pandas(args.output)
    print("Check import matches export.")
    env_t_copy = Preprocessing.from_pandas(df)

    if env_t != env_t_copy:
        sys.exit("Conversion to and from pandas did not yield the same result.")

    print("Success.")
