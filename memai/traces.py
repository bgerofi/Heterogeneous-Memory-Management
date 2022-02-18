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
            return float(
                self._data_["Timestamp"].iloc[-1] - self._data_["Timestamp"].iloc[0]
            )
        except IndexError:
            return 0.0

    def timestamps(self):
        self._data_["Timestamp"].values

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
        )
        win_end = np.searchsorted(
            self._data_["Timestamp"],
            self._data_["Timestamp"].iloc[0] + (window_len * (win + 1)),
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
            )
            return self._subset_(slice(win_begin, -1))
        else:
            win_begin = np.searchsorted(
                self._data_["Instrs"],
                self._data_["Instrs"].iloc[0] + (window_len * win),
            )
            win_end = np.searchsorted(
                self._data_["Instrs"],
                self._data_["Instrs"].iloc[0] + (window_len * (win + 1)),
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

    def __len__(self):
        return self._data_.__len__()

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
        window_length=50,
        compare_unit="ms",
        verbose=False,
    ):
        self.trace_ddr = trace_ddr
        self.trace_hbm = trace_hbm

        self.nr_win = 0
        self.window_length = 0
        self.window_length2 = 0
        self.compare_unit = compare_unit
        self.set_window_properties(window_length, compare_unit)

        self._verbose_ = verbose

    def set_window_properties(self, window_length, compare_unit):
        """
        Change the way an existing TraceSet is split into windows.
        This can be used if for some phases it is more meaningful to
        change the comparison unit or length of the windows.
        The window properties can be set with two knobs:
        window_length and the comparison unit which can be time wise (ms),
        instructions wise (instrs) or memory access wise (accesses).
        """
        if compare_unit == "accesses":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_access_bounds_
            self._subset_window_fn_ = Trace.subset_window_access
        elif compare_unit == "instrs":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_instruction_bounds_
            self._subset_window_fn_ = Trace.subset_window_instruction
        elif compare_unit == "ms":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_timestamp_bounds_
            self._subset_window_fn_ = Trace.subset_window_timestamp
        else:
            err_msg = "Invalid window comparison unit: {}.\n".format(compare_unit)
            err_msg += "Expected one of ['accesses', 'instrs', 'ms']"
            raise ValueError(err_msg)
        self.compare_unit = compare_unit
        self._set_window_(window_length)

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
        return self.nr_win == 0

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

    def _get_phase_(self, phase):
        """
        Build a new TraceSet from this TraceSet containing a single phase of it.
        """
        return Phase(phase, self)

    def _get_window_(self, win):
        """
        Build a new TraceSet from this TraceSet containing a single window of
        it. This is usually called when this TraceSet is made of a single phase.
        """
        if win >= self.nr_win:
            raise ValueError(
                "Window {} out of bounds " "({}).".format(win, self.nr_win)
            )
        trace_ddr = self._subset_window_fn_(
            self.trace_ddr, self.window_length, win, self.nr_win
        )
        trace_hbm = self._subset_window_fn_(
            self.trace_hbm, self.window_length2, win, self.nr_win
        )
        trace = TraceSet(
            trace_ddr,
            trace_hbm,
            self.window_length,
            self.compare_unit,
            self._verbose_,
        )
        return trace

    def print_info(self):
        print(
            "{} phases, {} windows, with window_length: {} {}, "
            "window_length2: {} {}, ddr_timespan: {} (ms), "
            "hbm timespan: {} (ms)".format(
                len(self.trace_ddr.phases()),
                self.nr_win,
                self.window_length,
                self.compare_unit,
                self.window_length2,
                self.compare_unit,
                self.trace_ddr.timespan(),
                self.trace_hbm.timespan(),
            )
        )

    def __iter__(self):
        return WindowIterator(self)

    def __len__(self):
        return len(self.trace_ddr)

    def _get_trace_timestamp_bounds_(trace):
        """
        Get bounds of the trace (minimum timestamp, maximum timestamp)
        in the "ms" unit.
        """
        return float(trace["Timestamp"].iloc[-1] - trace["Timestamp"].iloc[0])

    def _get_trace_instruction_bounds_(trace):
        """
        Get bounds of the trace (minimum instruction number,
        maximum instruction number) in the "instruction" unit.
        """
        return float(trace["Instrs"].iloc[-1] - trace["Instrs"].iloc[0])

    def _get_trace_access_bounds_(trace):
        """
        Get bounds of the trace (minimum memory access number,
        maximum memory access number) in the "access" unit.
        """
        return float(len(trace))

    def _set_window_(self, window_length):
        """
        Set number of windows, and window length for each trace according
        to the selected unit.
        """
        try:
            t_ddr = self._get_trace_bounds_fn_(self.trace_ddr)
        except IndexError:
            return

        window_length = float(window_length) if window_length is not None else t_ddr
        window_length = min(window_length, t_ddr)
        if window_length == 0:
            return

        self.nr_win = int(t_ddr / window_length)
        if self.nr_win == 0:
            self.nr_win = 1
        self.window_length = t_ddr / self.nr_win

        try:
            t_hbm = self._get_trace_bounds_fn_(self.trace_hbm)
        except IndexError:
            self.nr_win = 0
            return

        self.window_length2 = t_hbm / self.nr_win


class Phase(TraceSet):
    """
    A Phase is a subset of a TraceSet containing a single phase.
    """

    def __init__(self, phase, trace_set):
        trace_ddr = trace_set.trace_ddr.get_phases(phase)
        trace_hbm = trace_set.trace_hbm.get_phases(phase)
        compare_unit = trace_set.compare_unit
        window_length = trace_set.window_length

        super(Phase, self).__init__(trace_ddr, trace_hbm, window_length, compare_unit)
        self.phase = phase

    def print_info(self):
        print(
            "Phase {}, {} windows, with window_length: {} {}, "
            "window_length2: {} {}, ddr_timespan: {} (ms), "
            "hbm timespan: {} (ms)".format(
                self.phase,
                self.nr_win,
                self.window_length,
                self.compare_unit,
                self.window_length2,
                self.compare_unit,
                self.trace_ddr.timespan(),
                self.trace_hbm.timespan(),
            )
        )


class WindowIterator:
    """
    TraceSet iterator iterating through Phases of the traces and through windows
    inside phases of the trace. On each iteration, the next window is returned.
    """

    def __init__(self, trace_set):
        phases = trace_set.trace_ddr.phases()
        self._phases_ = iter(phases)
        self._traces_ = trace_set
        self._win_ = 0
        self._phase_ = None
        self._set_next_phase_()

    def _set_next_phase_(self):
        """
        Move iterator to next phase.
        """
        self._phase_ = self._traces_._get_phase_(next(self._phases_))
        self._win_ = 0

    def __next__(self):
        """
        Get next window of the traces.
        """
        try:
            window = self._phase_._get_window_(self._win_)
            self._win_ += 1
            return window
        except ValueError:
            self._set_next_phase_()
            return next(self)
