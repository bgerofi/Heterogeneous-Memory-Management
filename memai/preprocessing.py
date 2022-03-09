import re
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
    WINDOW = "Window"

    def __init__(
        self,
        observation_space=WindowObservationSpace(128, 128),
        page_size=20,
        interval_distance=28,
    ):
        self.observation_space = observation_space
        self.observations = []
        self.pages = []
        self.accesses = []
        self.t_ddr = []
        self.t_hbm = []
        self.windows = []
        self.intervals = IntervalDetector(1 << interval_distance, 1 << page_size)
        self._window_index = 0

    def __iter__(self):
        return zip(
            self.observations,
            self.pages,
            self.accesses,
            self.t_ddr,
            self.t_hbm,
            self.windows,
        )

    def _append_(self, observation, pages, accesses, t_ddr, t_hbm):
        self.observations.append(observation)
        self.pages.append(pages)
        self.accesses.append(accesses)
        self.t_ddr.append(t_ddr)
        self.t_hbm.append(t_hbm)
        self.windows.append(self._window_index)

    def _append_window_(self, window):
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

        self._window_index += 1

    def as_dict(self):
        return {
            Preprocessing.OBSERVATION: [o.flatten() for o in self.observations],
            Preprocessing.PAGES: self.pages,
            Preprocessing.ACCESSES: self.accesses,
            Preprocessing.TIME_DDR: self.t_ddr,
            Preprocessing.TIME_HBM: self.t_hbm,
            Preprocessing.WINDOW: self.windows,
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
        pre = Preprocessing(observation_space, page_size, interval_distance)
        windows = WindowIterator(trace_set, compare_unit, window_len)
        progress_bar = tqdm.tqdm()
        last_timestamp = int(windows._bounds_fn_(trace_set.trace_ddr)[1])
        progress_bar.reset(total=last_timestamp)

        for window in windows:
            pre._append_window_(window)
            progress_bar.update(int(window.t_ddr))

        progress_bar.update(last_timestamp)
        progress_bar.clear()
        return pre

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
            Preprocessing.WINDOW,
        ]
        if not all(df.columns == colnames):
            raise ValueError(
                "Invalid dataframe columns name. Expected one of: {}".format(colnames)
            )
        if len(df) == 0:
            raise ValueError("Empty traces not supported.")

        pre = Preprocessing(observation_space, page_size, interval_distance)
        pre.observations = [
            o.reshape(observation_space.shape)
            for o in df[Preprocessing.OBSERVATION].values
        ]
        pre.pages = df[Preprocessing.PAGES].values
        pre.accesses = df[Preprocessing.ACCESSES].values
        pre.t_ddr = df[Preprocessing.TIME_DDR].values
        pre.t_hbm = df[Preprocessing.TIME_HBM].values
        pre.windows = df[Preprocessing.WINDOW].values
        pre.intervals.append_addresses(np.unique(np.concatenate(pre.pages)))
        return pre

    @staticmethod
    def parse_filename(filename):
        interval_distance = re.findall("interval_distance=(?P<dist>[0-9]+)", filename)
        if len(interval_distance) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        interval_distance = int(interval_distance[0])

        window_len = re.findall("window_len=(?P<dist>[0-9]+)", filename)
        if len(window_len) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        window_len = int(window_len[0])

        page_size = re.findall("page_size=(?P<dist>[0-9]+)", filename)
        if len(page_size) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        page_size = int(page_size[0])

        observation_rows = re.findall("observation_rows=(?P<dist>[0-9]+)", filename)
        if len(observation_rows) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        observation_rows = int(observation_rows[0])

        observation_columns = re.findall(
            "observation_columns=(?P<dist>[0-9]+)", filename
        )
        if len(observation_columns) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        observation_columns = int(observation_columns[0])

        compare_unit = re.findall("compare_unit=(?P<dist>[A-Za-z]+)", filename)
        if len(compare_unit) == 0:
            raise ValueError("Filename does not contain 'interval_distance' value.")
        compare_unit = compare_unit[0]

        return {
            "interval_distance": interval_distance,
            "window_len": window_len,
            "page_size": page_size,
            "compare_unit": compare_unit,
            "observation_space": WindowObservationSpace(
                observation_rows, observation_columns
            ),
        }

    @staticmethod
    def make_filename(
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


if __name__ == "__main__":
    import argparse
    from memai import Trace, TraceSet
    from memai.options import *
    import sys
    import os

    parser = argparse.ArgumentParser()
    add_traces_input_args(parser)
    add_window_args(parser)
    add_observation_args(parser)
    add_interval_args(parser)

    args = parser.parse_args()

    output = Preprocessing.make_filename(
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
    pre = Preprocessing.from_trace_set(
        traces,
        args.page_size,
        args.interval_distance,
        args.compare_unit,
        args.window_len,
        observation_space,
    )
    print("Export processed traces to: {}".format(output))
    df = pret.as_pandas(output)
