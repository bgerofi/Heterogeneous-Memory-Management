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

"""
This module defines the class used to preprocess a bundle of two
traces (TraceSet) of memory access from an application into a trace of
windows' observations usable by a RL algorithm and associated page accesses
and execution times in ddr and hbm used to evaluate execution time of the
window with arbitrary data mapping in HBM.

This files is also a script to perform this preprocessing and store their
resulting dataframe for later use.

See `memai.observation.WindowObservationSpace`
See `memai.traces.TraceSet`
See `memai.estimation.Estimator`
"""


class Preprocessing:
    """
    Class representing a set of preprocessed windows of observations, page
    accesses, execution time in ddr, execution time in hbm.
    Items of this trace are further fed as input to the RL algorithm and
    trace estimator to give a feedback to the RL algorithm.

    This class is instanciated as empty and is built iteratively window by
    window from a `TraceSet`. This whole iteration process is implemented
    by the constructor method: `from_trace_set()`.

    Observations of this class are a 2 dimension matrix where columns
    represent timestamps and rows represent addresses.
    Since the space of addresses accessed by the application can be very large
    in address values but sparse and narrow in contiguous address ranges,
    windows are further subdivided into smaller contiguous regions of
    addresses actually accessed by the application and having their own
    memory access pattern. Therefore their can be multiple observations,
    per window in the final preprocessed trace, up to one per accessed
    contiguous address range.

    This set of contiguous regions is also built incrementally as follow:
    For each window, the accessed address ranges are added into an interval
    tree. Added addresses are extended to the left and to the right by a fixed
    interval distance. When address ranges in the interval tree overlap,
    they are merged into a single address range. The set of address ranges
    in the intervaltree after the current window has been processed defines
    the maximum number of possible observations for the window. If an address
    range is not accessed by a window it is not added as an observation.
    """

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
        """
        Preprocessing constructor, initialized as empty.
        @arg observation_space: The observation space used to create
        observations from a list of (address, timestamp).
        See `memai.observation.WindowObservationSpace`.
        @arg page_size: The exponent (of a power of two) defining the size of
        pages. Pages are used as a unit of access. Accessed addresses are
        converted to a page range where the beginning of the page is the
        closest smaller address aligned on a page boundary.
        @arg: interval_distance: The exponent (of a power of two) of the size
        to append to the left and to the right of addresses accessed to
        compute the interval tree of accessed memory ranges.
        See above class description.
        """

        self.observation_space = observation_space
        self.observations = []
        self.pages = []
        self.accesses = []
        self.t_ddr = []
        self.t_hbm = []
        self.windows = []
        self.intervals = IntervalDetector(1 << interval_distance, 1 << page_size)
        self._window_index = 0

    def __len__(self):
        return len(self.observations)

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
        """
        Update intervaltree of accessed contiguous address ranges.
        Split a window into accesses in the accessed contiguous address ranges.
        For each access range, create an observation and store it along with
        accessed pages, access count and timing of the window in ddr and hbm.
        """
        # IF the window is empty add an empty observation and return.
        if window.is_empty():
            self._append_(
                self.observation_space.empty(), [], [], window.t_ddr, window.t_ddr
            )
            self._window_index += 1
            return

        # Collect page addresses of accesses and update intervaltree of
        # accessed contiguous regions.
        window_pages = window.addresses & self.intervals.page_mask
        self.intervals.append_addresses(window_pages)

        # Iterate through regions of addresses and if they contain accesses
        # for the current window, create an observation.
        for interval in self.intervals:
            indexes = [interval.contains_point(p) for p in window.addresses]
            if not any(indexes):
                continue

            is_empty = False
            addr = window.addresses[indexes]
            timestamps = window.traces.trace_ddr[Trace.TIMESTAMP][indexes]
            page, count = np.unique(addr & self.intervals.page_mask, return_counts=True)
            observation = self.observation_space.from_sparse_matrix(
                timestamps, addr >> self.intervals.page_shift
            )
            self._append_(observation, page, count, window.t_ddr, window.t_hbm)
        self._window_index += 1

    def as_dict(self):
        """
        Convert the current state of the Preprocessing object into a
        dictionnary.
        This is mainly used as an intermediate state to convert this object
        to a pandas dataframe.
        """
        return {
            Preprocessing.OBSERVATION: [o.flatten() for o in self.observations],
            Preprocessing.PAGES: self.pages,
            Preprocessing.ACCESSES: self.accesses,
            Preprocessing.TIME_DDR: self.t_ddr,
            Preprocessing.TIME_HBM: self.t_hbm,
            Preprocessing.WINDOW: self.windows,
        }

    def as_pandas(self, output_file=None):
        """
        Convert the current state of the Preprocessing object into a
        pandas dataframe.
        """
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
        """
        Preprocess a bundle of execution traces (ddr + hbm) into a
        preprocessing object that can be iterated or stored into a new
        dataframe.
        """
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
        """
        Read a pandas dataframe of an already preprocessed bundle of traces
        into a Preprocessing object that can be iterated in a gym environmnent.
        """

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
        # pre.intervals.append_addresses(np.unique(np.concatenate(pre.pages)))
        return pre

    @staticmethod
    def parse_filename(filename):
        """
        Parse the name of a preprocessed trace to retrieve the static
        parameters of the preprocessing: interval_distance, window_length,
        page_size, window_unit, observation_rows, observation_columns.
        """

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
        """
        Create a filename containing the static
        parameters of a Preprocessing object: interval_distance,
        window_length, page_size, window_unit, observation_rows,
        observation_columns.
        """
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
    df = pre.as_pandas(output)
