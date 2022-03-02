import pandas as pd
import numpy as np
from intervaltree import IntervalTree


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
        else:
            # Make instructions as an increasing number.
            for phase in range(1, data["Phase"].iloc[-1]):
                i = np.searchsorted(data["Phase"], phase)
                if i == 0:
                    continue
                data.loc[i:, "Instrs"] += data["Instrs"][i - 1]

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

    def num_instructions(self):
        """
        Get bounds of the trace (minimum instruction number,
        maximum instruction number) in the "instruction" unit.
        """
        try:
            return self._data_["Instrs"].iloc[-1] - self._data_["Instrs"].iloc[0]
        except IndexError:
            return 0

    def __len__(self):
        return self._data_.__len__()

    def timestamps(self):
        return self._data_["Timestamp"].values

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

    def get_phases(self, phase):
        """
        Build a trace from this trace containing only the selected phases.
        If `phase` is an int, then only one phase is selected.
        If `phase` is a list of int, then all the phases it contains are
        selected.
        An error will be raised if a phase is not present in the trace.
        """
        if isinstance(phase, int) or isinstance(phase, np.int64):
            phase = [phase]

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

    def subset_window_timestamp(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the "Timestamp" column of the pandas data
        frame"
        """
        win_begin = np.searchsorted(
            self._data_["Timestamp"],
            self._data_["Timestamp"].iloc[0] + (window_len * win),
            side="left",
        )
        win_end = np.searchsorted(
            self._data_["Timestamp"],
            self._data_["Timestamp"].iloc[0] + (window_len * (win + 1)),
            side="right",
        )
        return self._subset_(slice(win_begin, win_end))

    def subset_window_instruction(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the "Instrs" column of the pandas data
        frame"
        """
        if win == nr_wins - 1:
            win_begin = np.searchsorted(
                self._data_["Instrs"],
                self._data_["Instrs"].iloc[0] + (window_len * win),
                side="left",
            )
            return self._subset_(slice(win_begin, -1))
        else:
            win_begin = np.searchsorted(
                self._data_["Instrs"],
                self._data_["Instrs"].iloc[0] + (window_len * win),
                side="left",
            )
            win_end = np.searchsorted(
                self._data_["Instrs"],
                self._data_["Instrs"].iloc[0] + (window_len * (win + 1)),
                side="right",
            )
            return self._subset_(slice(win_begin, win_end))

    def subset_window_access(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the number of samples in the pandas data
        frame"
        """
        if win == nr_wins - 1:
            return self._subset_(slice(int(window_len * win), -1))
        else:
            return self._subset_(
                slice(int(window_len * win), int(window_len * (win + 1)))
            )

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
        if self.is_empty():
            return "Empty window."
        s = "DDR: [{:.8g} - {:.8g}](ms) -- HBM: [{:.8g} - {:.8g}](ms)".format(
            self.time_start(),
            self.time_end(),
            self.trace_hbm["Timestamp"].iloc[0],
            self.trace_ddr["Timestamp"].iloc[-1],
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
        return len(self.trace_ddr) == 0 or len(self.trace_hbm) == 0

    def time_start(self):
        return self.trace_ddr["Timestamp"].iloc[0]

    def time_end(self):
        return self.trace_ddr["Timestamp"].iloc[-1]

    def timespan_ddr(self):
        """
        Get the total time in the ddr trace.
        """
        return self.trace_ddr.timespan()

    def timespan_hbm(self):
        """
        Get the total time in the hbm trace.
        """
        return self.trace_hbm.timespan()

    def subset_phases(self, phases):
        """
        Filter this TraceSet to contain only selected phases.
        """
        self.trace_ddr = self.trace_ddr.get_phases(phases)
        self.trace_hbm = self.trace_hbm.get_phases(phases)

    def __str__(self):
        print(
            "{} phases. ddr_timespan: {} (ms), hbm timespan: {} (ms)".format(
                len(self.trace_ddr.phases()),
                self.trace_ddr.timespan(),
                self.trace_hbm.timespan(),
            )
        )

    def __len__(self):
        return len(self.trace_ddr)


class WindowIterator:
    """
    TraceSet iterator iterating through Phases of the traces and through windows
    inside phases of the trace. On each iteration, the next window is returned.
    """

    def __init__(self, trace_set, compare_unit="ms", window_length=50):
        self._traces = trace_set
        self._phases = iter(trace_set.trace_ddr.phases())
        self._target_win_len = float(window_length)
        self._win = iter([])

        if compare_unit == "accesses":
            self._get_trace_bounds_fn_ = Trace.__len__
            self._subset_window_fn_ = Trace.subset_window_access
        elif compare_unit == "instrs":
            self._get_trace_bounds_fn_ = Trace.num_instructions
            self._subset_window_fn_ = Trace.subset_window_instruction
        elif compare_unit == "ms":
            self._get_trace_bounds_fn_ = Trace.timespan
            self._subset_window_fn_ = Trace.subset_window_timestamp

    def next_phase(self):
        """
        Move iterator to next phase.
        """
        phase = next(self._phases)
        trace_ddr = self._traces.trace_ddr.get_phases(phase)
        trace_hbm = self._traces.trace_hbm.get_phases(phase)
        phase = TraceSet(trace_ddr, trace_hbm)
        t_hbm = self._get_trace_bounds_fn_(trace_hbm)
        t_ddr = self._get_trace_bounds_fn_(trace_ddr)
        nr_win = int(t_ddr / self._target_win_len)
        if t_ddr < self._target_win_len:
            return phase

        self._win_index_stop = 0
        self._ddr_win_len = t_ddr / nr_win
        self._hbm_win_len = t_hbm / nr_win
        self._win = iter(range(nr_win))
        self._nr_win = nr_win
        self._phase = phase

        return next(self)

    def __next__(self):
        """
        Get next window of the phase. If the phase is ending, move to next phase.
        """
        try:
            win = next(self._win)
            ddr_win = self._subset_window_fn_(
                self._phase.trace_ddr, self._ddr_win_len, win, self._nr_win
            )
            hbm_win = self._subset_window_fn_(
                self._phase.trace_hbm, self._hbm_win_len, win, self._nr_win
            )
            window = TraceSet(ddr_win, hbm_win)
        except StopIteration:
            window = self.next_phase()

        return window

    def __iter__(self):
        return self
