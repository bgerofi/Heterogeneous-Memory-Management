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
    PAGES = "Pages"
    ACCESSES = "Accesses"
    TIME_DDR = "TimeDDR"
    TIME_HBM = "TimeHBM"

    def __init__(
        self,
        observation_space=WindowObservationSpace(128, 128),
        page_size=2 ^ 20,
        interval_distance=2 ^ 28,
    ):
        self.observation_space = observation_space
        self.observations = []
        self.pages = []
        self.accesses = []
        self.t_ddr = []
        self.t_hbm = []
        self.intervals = IntervalDetector(1 << interval_distance, 1 << page_size)

    def __iter__(self):
        return zip(self.observations, self.pages, self.accesses, self.t_ddr, self.t_hbm)

    def _append_(self, observation, pages, accesses, t_ddr, t_hbm):
        self.observations.append(observation)
        self.pages.append(pages)
        self.accesses.append(accesses)
        self.t_ddr.append(t_ddr)
        self.t_hbm.append(t_hbm)

    def __eq__(self, other):
        if not all(np.array(self.t_hbm) == np.array(other.t_hbm)):
            return False
        if not all(np.array(self.t_ddr) == np.array(other.t_ddr)):
            return False

        for self_obs, other_obs in zip(self.observations, other.observations):
            if not all(self_obs.flatten() == other_obs.flatten()):
                return False

        if not all(np.concatenate(self.accesses) == np.concatenate(other.accesses)):
            return False
        if not all(np.concatenate(self.pages) == np.concatenate(other.pages)):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def append_window(self, window):
        if window.is_empty():
            self._append_(
                self.observation_space.empty(), [], [], window.t_ddr, window.t_hbm
            )
            return

        window_pages = window.addresses & self.intervals.page_mask
        self.intervals.append_addresses(window_pages)

        for interval in self.intervals:
            indexes = [interval.contains_point(p) for p in window.addresses]
            if not any(indexes):
                continue

            addr = window.addresses[indexes]
            timestamps = window.traces.trace_ddr[Trace.TIMESTAMP][indexes]
            page, count = np.unique(addr & self.intervals.page_mask, return_counts=True)
            observation = self.observation_space.from_sparse_matrix(timestamps, addr)
            self._append_(observation, page, count, window.t_ddr, window.t_hbm)

    def as_dict(self):
        return {
            Preprocessing.OBSERVATION: [o.flatten() for o in self.observations],
            Preprocessing.PAGES: self.pages,
            Preprocessing.ACCESSES: self.accesses,
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
        env_t = Preprocessing(observation_space, page_size, interval_distance)
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
    def from_pandas(
        filename,
        page_size=1 << 20,
        interval_distance=1 << 28,
        observation_space=WindowObservationSpace(128, 128),
    ):
        if isinstance(filename, pd.DataFrame):
            df = filename
        elif isinstance(filename, str):
            df = pd.read_feather(filename)
        else:
            raise ValueError("Expected either a pandas dataframe or a filename.")
        colnames = [
            Preprocessing.OBSERVATION,
            Preprocessing.PAGES,
            Preprocessing.ACCESSES,
            Preprocessing.TIME_DDR,
            Preprocessing.TIME_HBM,
        ]
        if not all(df.columns == colnames):
            raise ValueError(
                "Invalid dataframe columns name. Expected one of: {}".format(colnames)
            )
        if len(df) == 0:
            raise ValueError("Empty traces not supported.")

        env_t = Preprocessing(observation_space, page_size, interval_distance)
        env_t.observations = [
            o.reshape(observation_space.shape)
            for o in df[Preprocessing.OBSERVATION].values
        ]
        env_t.pages = df[Preprocessing.PAGES].values
        env_t.accesses = df[Preprocessing.ACCESSES].values
        env_t.t_ddr = df[Preprocessing.TIME_DDR].values
        env_t.t_hbm = df[Preprocessing.TIME_HBM].values
        env_t.intervals.append_addresses(np.unique(np.concatenate(env_t.pages)))
        return env_t


if __name__ == "__main__":
    import argparse
    from memai import Trace, TraceSet
    from memai.options import *
    import sys
    import os

    def make_output(
        ddr_filename,
        window_len,
        compare_unit,
        interval_distance,
        page_size,
        observation_rows,
        observation_columns,
    ):
        f = ddr_filename.strip(".feather")
        f = f.split("-")
        f = [s for s in f if s not in ["DRAM", "MCDRAM"]]
        f.append("window_len={}".format(window_len))
        f.append(
            "compare_unit={}".format(
                compare_unit,
            )
        )
        f.append("interval_distance={}".format(interval_distance))
        f.append("page_size={}".format(page_size))
        f.append("observation_rows={}".format(observation_rows))
        f.append("observation_columns={}".format(observation_columns))
        return "-".join(f) + ".feather"

    parser = argparse.ArgumentParser()
    add_traces_input_args(parser)
    add_window_args(parser)
    add_observation_args(parser)
    add_interval_args(parser)

    parser.add_argument(
        "--output",
        metavar="<file>",
        type=str,
        default=None,
        help=("Where to write the obtained pandas dataframe."),
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = make_output(
            args.ddr_input,
            args.window_len,
            args.compare_unit,
            args.interval_distance,
            args.page_size,
            args.observation_rows,
            args.observation_columns,
        )

    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms)
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
    )
    if args.interval_distance <= args.page_size:
        raise ValueError("Interval distance must be greater than page size.")

    print("Processing traces.")
    observation_space = WindowObservationSpace(
        args.observation_rows, args.observation_columns
    )
    env_t = Preprocessing.from_trace_set(
        traces,
        args.page_size,
        args.interval_distance,
        args.compare_unit,
        args.window_len,
        observation_space,
    )

    print("Export processed traces to: {}".format(args.output))
    df = env_t.as_pandas(args.output)
    print("Check import matches export.")
    env_t_copy = Preprocessing.from_pandas(
        df,
        args.page_size,
        args.interval_distance,
        observation_space,
    )

    if env_t != env_t_copy:
        raise ValueError("Conversion to and from pandas did not yield the same result.")

    print("Success.")
