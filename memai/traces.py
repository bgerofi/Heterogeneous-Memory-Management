import pandas as pd
import numpy as np


class Trace:
    """
    Class representing a single trace file.
    The file contains one or several phases and can subseted to one of these.
    It also be split into windows and subseted to one of these. The window
    properties are defined by the user, and the metric used for for subsetting
    is also to be chosen.
    """

    def __init__(self, filename, cpu_cycles_per_ms, verbose=False):
        data = pd.read_feather(filename)

        # Make Timestamp as milliseconds
        data["Timestamp"] = data["Timestamp"].div(cpu_cycles_per_ms)
        if "Phase" not in data.columns:
            data = data.assign(Phase=0)

        # Make instructions as an increasing number.
        data = data.assign(
            Instrs=np.cumsum(data.Instrs.values), Index=list(range(len(data)))
        )

        self._data_ = data
        self.filename = filename

        if verbose:
            self.print_info()

    def is_empty(self):
        """
        Return whether there is no sample in the trace.
        """
        return len(self._data_) == 0

    def singlify_phases(self):
        """
        Set the trace such that there is only one phase in it.
        """
        self._data_ = self._data_.assign(Phase=0)

    def phases(self):
        """
        Get a list of phases in the trace.
        """
        try:
            return list(
                range(self._data_["Phase"].iloc[0], self._data_["Phase"].iloc[-1] + 1)
            )
        except IndexError:
            return []

    def timespan(self):
        """
        Return the timespan betwenn first and last sample
        """
        try:
            return self._data_["Timestamp"].iloc[-1] - self._data_["Timestamp"].iloc[0]
        except IndexError:
            return 0

    @staticmethod
    def time_range(dataframe):
        return dataframe["Timestamp"].iloc[0], dataframe["Timestamp"].iloc[-1]

    @staticmethod
    def instruction_range(dataframe):
        return dataframe["Instrs"].iloc[0], dataframe["Instrs"].iloc[-1]

    @staticmethod
    def access_range(dataframe):
        return dataframe["Index"].iloc[0], dataframe["Index"].iloc[-1]

    def __len__(self):
        return self._data_.__len__()

    def virtual_addresses(self):
        """
        Get the virtual address of each sample in the trace.
        """
        return self._data_["Vaddr"].values

    def print_info(self):
        if self.is_empty():
            print("Empty trace: {}.".format(self.filename))
        else:
            print(
                "Loaded {} accesses ({} phases) between {} and {} "
                "msecs".format(
                    len(self._data_),
                    len(self.phases()),
                    self._data_["Timestamp"].iloc[0],
                    self._data_["Timestamp"].iloc[-1],
                )
            )

    def _subset_(self, select):
        """
        Build a new trace with a subset of this trace.
        Element in the pandas data frame of this trace are selected passsing
        `select` argument to the `__getitem__()` method of the pandas data
        frame.
        """
        t = Trace.__new__(Trace)
        t.filename = self.filename
        t._data_ = self._data_[select]
        return t

    def get_phase(self, phase):
        phase_start = np.searchsorted(self._data_["Phase"], phase, side="left")
        phase_stop = np.searchsorted(self._data_["Phase"], phase + 1, side="right")
        return self._subset_(slice(phase_start, phase_stop))

    def get_phases(self, phase):
        """
        Build a trace from this trace containing only the selected phases.
        If `phase` is an int, then only one phase is selected.
        If `phase` is a list of int, then all the phases it contains are
        selected.
        An error will be raised if a phase is not present in the trace.
        """
        if isinstance(phase, int) or isinstance(phase, np.int64):
            return self.get_phase(phase)

        available_phases = self.phases()
        if any((p not in available_phases for p in phase)):
            raise ValueError(
                "Invalid phase {}. Possible values are: {}".format(
                    phase, available_phases
                )
            )
        if len(phase) == 1:
            return self._subset_(self._data_.Phase == phase[0])
        else:
            return self._subset_([p in phase for p in self._data_.Phase])

    def timestamps_data(self):
        return self._data_.Timestamp

    def instructions_data(self):
        return self._data_.Instrs

    def accesses_data(self):
        return self._data_.Index

    def __getitem__(self, item):
        return self._data_.__getitem__(item)


class TraceSet:
    """
    Make a bundle of two traces where the first trace is the trace
    containing the execution of a single application in DDR memory, the
    second trace is the trace containing the execution of the same application
    in High Bandwidth Memory (hbm)

    This trace set, iterated with a window iterator, i.e an iterator
    returning a contiguouus subset of elements of the three traces representing
    the execution of the same region of an application in the two different
    memory set ups.

    The window properties can be set with two knobs:
    window_length and the comparison unit which can be time wise (ms),
    instructions wise (instrs) or memory access wise (accesses).
    """

    def __init__(
        self,
        trace_ddr,
        trace_hbm,
    ):
        self.trace_ddr = trace_ddr
        self.trace_hbm = trace_hbm

    def __str__(self):
        s = "DDR: "
        if len(self.trace_ddr) == 0:
            s += "[{:^21}](ms)".format("empty")
        else:
            s += "[{:8g} - {:8g}](ms)".format(
                self.trace_ddr["Timestamp"].iloc[0],
                self.trace_ddr["Timestamp"].iloc[-1],
            )
        s += " -- HBM: "
        if len(self.trace_hbm) == 0:
            s += "[{:^19}](ms)".format("empty")
        else:
            s += "[{:8g} - {:8g}](ms)".format(
                self.trace_hbm["Timestamp"].iloc[0],
                self.trace_hbm["Timestamp"].iloc[-1],
            )
        return s

    def singlify_phases(self):
        """
        Set all traces in this TraceSet such that there is only the same single
        phase in them.
        """
        self.trace_ddr.singlify_phases()
        self.trace_hbm.singlify_phases()

    def is_empty(self):
        """
        Return whether there is no sample in this TraceSet.
        """
        return min(len(self.trace_ddr), len(self.trace_hbm)) == 0

    def time_end(self):
        return self.trace_ddr["Timestamp"].iloc[-1]

    def subset_phases(self, phases):
        """
        Filter this TraceSet to contain only selected phases.
        """
        self.trace_ddr = self.trace_ddr.get_phases(phases)
        self.trace_hbm = self.trace_hbm.get_phases(phases)

    def __len__(self):
        return len(self.trace_ddr)


class WindowItem:
    def __init__(
        self,
        t_ddr_begin,
        t_ddr_end,
        t_hbm_begin,
        t_hbm_end,
        addresses=[],
    ):
        self.addresses = addresses
        self.t_ddr_begin = t_ddr_begin
        self.t_ddr_end = t_ddr_end
        self.t_hbm_begin = t_hbm_begin
        self.t_hbm_end = t_hbm_end

    def __str__(self):
        s = "DDR: [{:8g} - {:8g}](ms)".format(
            self.t_ddr_begin,
            self.t_ddr_end,
        )
        s += " -- HBM: [{:8g} - {:8g}](ms)".format(
            self.t_hbm_begin,
            self.t_hbm_end,
        )
        if not self.is_empty():
            s += "\nAddresses: {} [{}:{}]".format(
                len(self.addresses), hex(min(self.addresses)), hex(max(self.addresses))
            )
        return s

    @property
    def t_ddr(self):
        return self.t_ddr_end - self.t_ddr_begin

    @property
    def t_hbm(self):
        return self.t_hbm_end - self.t_hbm_begin

    def is_empty(self):
        return len(self.addresses) == 0


class WindowIterator:
    """
    TraceSet iterator iterating through Phases of the traces and through windows
    inside phases of the trace. On each iteration, the next window is returned.
    """

    def __init__(self, trace_set, compare_unit="ms", window_length=None):
        self._traces = trace_set
        self._phases = iter(trace_set.trace_ddr.phases())
        self._windows = iter([])
        self._bounds_fn_, self._timestamp_fn_ = WindowIterator._parse_unit(compare_unit)
        self._last_timestamp_ddr, te = self._bounds_fn_(trace_set.trace_ddr)
        self._last_timestamp_hbm, _ = self._bounds_fn_(trace_set.trace_hbm)

        if window_length is None:
            num_phases = len(trace_set.trace_ddr.phases())
            window_length = (te - self._last_timestamp_ddr) / num_phases
            window_length = window_length
            window_length = max(1, window_length)
            widow_length = min(te - self._last_timestamp_ddr, window_length)
        self._target_win_len = float(window_length)

    @staticmethod
    def _parse_unit(unit):
        if unit == "accesses":
            return Trace.access_range, Trace.accesses_data
        elif unit == "instrs":
            return Trace.instruction_range, Trace.instructions_data
        elif unit == "ms":
            return Trace.time_range, Trace.timestamps_data
        else:
            raise ValueError(
                "Invalid unit: {}. Must be one of: 'ms', 'instrs', 'accesses'.",
                format(unit),
            )

    def subset_window(self, df, begin, end):
        timestamps = self._timestamp_fn_(df)
        begin = np.searchsorted(timestamps, begin, side="left")
        if end != -1:
            end = np.searchsorted(timestamps, end, side="right")
        return df[begin:end]

    def next_phase(self):
        """
        Move iterator to next phase.
        """
        while True:
            phase = next(self._phases)
            trace_ddr = self._traces.trace_ddr.get_phases(phase)
            trace_hbm = self._traces.trace_hbm.get_phases(phase)
            if len(trace_ddr) > 0:
                break

        ddr_s, ddr_e = self._bounds_fn_(trace_ddr)
        t_ddr = ddr_e - ddr_s
        try:
            hbm_s, hbm_e = self._bounds_fn_(trace_hbm)
            t_hbm = hbm_e - hbm_s
        except IndexError:
            t_hbm = 0

        if t_ddr < self._target_win_len:
            nr_win = 1
            win_len_ddr = t_ddr
            win_len_hbm = t_hbm
        else:
            nr_win = int(t_ddr / self._target_win_len)
            win_len_ddr = t_ddr / nr_win
            win_len_hbm = t_hbm / nr_win
        win_len_ddr = max(1, win_len_ddr)

        windows = []
        while self._last_timestamp_ddr < ddr_e:
            t_hbm_begin = self._last_timestamp_hbm
            t_ddr_begin = self._last_timestamp_ddr
            t_hbm_end = self._last_timestamp_hbm + win_len_hbm
            t_ddr_end = self._last_timestamp_ddr + win_len_ddr
            hbm_win = self.subset_window(trace_hbm, t_hbm_begin, t_hbm_end)
            ddr_win = self.subset_window(trace_ddr, t_ddr_begin, t_ddr_end)
            addresses = []

            if len(hbm_win) > 0:
                _, t_end = self._bounds_fn_(hbm_win)
                t_hbm_end = max(t_hbm_end, t_end)
                t_hbm_end = min(t_hbm_end, hbm_e)
            if len(ddr_win) > 0:
                _, t_end = self._bounds_fn_(ddr_win)
                t_ddr_end = max(t_ddr_end, t_end)
                t_ddr_end = min(t_ddr_end, ddr_e)
                addresses = ddr_win["Vaddr"].values

            self._last_timestamp_hbm = t_hbm_end
            self._last_timestamp_ddr = t_ddr_end

            windows.append(
                WindowItem(
                    t_ddr_begin,
                    t_ddr_end,
                    t_hbm_begin,
                    t_hbm_end,
                    addresses,
                )
            )
        self._windows = iter(windows)

        return next(self)

    def __next__(self):
        """
        Get next window of the phase. If the phase is ending, move to next phase.
        """
        try:
            return next(self._windows)
        except StopIteration:
            return self.next_phase()

    def __iter__(self):
        return self
