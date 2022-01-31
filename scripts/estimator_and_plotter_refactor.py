# /usr/bin/python
import os
import pandas as pd
import numpy as np
import argparse
from intervaltree import Interval, IntervalTree


class Trace:
    """
    Class representing a single trace file.
    The file contains one or several phases and can subseted to one of these.
    It also be split into windows and subseted to one of these. The window
    properties are defined by the user, and the metric used for for subsetting
    is also to be chosen.
    """

    def __init__(self, filename, cpu_cycles_per_ms, verbose=False):
        if verbose:
            print("Loading from {}...".format(filename))

        # Private
        data = pd.read_feather(filename)
        data["Timestamp"] = data["Timestamp"].div(cpu_cycles_per_ms)
        if "Phase" not in data.columns:
            data = data.assign(Phase=0)

        self._data_ = data
        # Public
        self.filename = filename

        if verbose:
            self.print_info()

    def is_empty(self):
        """
        Return whether there is no sample in the trace.
        """
        return len(self_data_) == 0

    def singlify_phases(self):
        """
        Set the trace such that there is only one phase in it.
        """
        self._data_ = self._data_.assign(Phase=0)

    def phases(self):
        """
        Get a list of phases in the trace.
        """
        return np.unique(self._data_["Phase"])

    def virtual_addresses(self):
        """
        Get the virtual address of each sample in the trace.
        """
        return self._data_["Vaddr"].values

    def print_info(self):
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
        t.data = self._data_[select]
        return t

    def get_phases(self, phase):
        """
        Build a trace from this trace containing only the selected phases.
        If `phase` is an int, then only one phase is selected.
        If `phase` is a list of int, then all the phases it contains are
        selected.
        An error will be raised if a phase is not present in the trace.
        """
        if isinstance(phase, int):
            phase = [phase]
        if not isinstance(phase, list):
            raise TypeError(
                "Provided phase must be either a number or a list " "of numbers."
            )

        available_phases = self.phases()
        if any((p not in available_phases for p in phases)):
            raise ValueError(
                "Invalid phase {}. Possible values are: {}".format(
                    phase, available_phases
                )
            )
        return self._subset_(self._data_.Phase in phase)

    def subset_window_timestamp(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the "Timestamp" column of the pandas data
        frame"
        """
        return self._subset_(
            (self._data_["Timestamp"] >= (window_len * win))
            & (self._data_["Timestamp"] < (window_len * (win + 1)))
        )

    def subset_window_instruction(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the "Instrs" column of the pandas data
        frame"
        """
        if win == nr_wins - 1:
            return self._subset_(self._data_["Instrs"] >= (window_len * win))
        else:
            return self._subset_(
                (self._data_["Instrs"] >= (window_len * win))
                & (self._data_["Instrs"] < (window_len * (win + 1)))
            )

    def subset_window_access(self, window_len, win, nr_wins):
        """
        Build a trace from this trace with a subset of this trace element
        forming a window based on the number of samples in the pandas data
        frame"
        """
        if win == nr_wins - 1:
            return self._subset_(slice(window_len * win))
        else:
            return self._subset_(slice(compare_window * win), (window_len * (win + 1)))

    def print_phase_info(self, phase):
        print(
            "{}: {} accesses in phase {}, "
            "time window between {} and {} ({}) msecs, "
            "retired instructions: {}, mem/instr: {}".format(
                self._data_.filename,
                len(self._data_),
                phase,
                self._data_["Timestamp"].iloc[0],
                self._data_["Timestamp"].iloc[-1],
                self._data_["Timestamp"].iloc[-1] - self._data_["Timestamp"].iloc[0],
                self._data_["Instrs"].iloc[-1],
                len(self._data_) * 8 / trace["Instrs"].iloc[-1],
            )
        )

    def __len__(self):
        return self._data_.__len__()

    def __getitem__(self, item):
        return self._data_.__getitem__(item)


class TraceSet:
    """
    Make a bundle of three traces where the first trace is the trace
    containing the execution of a single application in DDR memory, the
    second trace is the trace containing the execution of the same application
    in High Bandwidth Memory (hbm), and the last trace contains an execution of
    the same application with an arbitrary memory allocation.
    (The last trace may be `None`.)

    This trace set, iterated with a window iterator, i.e an iterator
    returning a contiguouus subset of elements of the three traces representing
    the execution of the same region of an application in three different
    memory set up.

    The window properties can be set with two knobs:
    window_length and the comparison unit which can be time wise (ms),
    instructions wise (instrs) or memory access wise (accesses).
    """

    def __init__(
        self,
        trace_ddr,
        trace_hbm,
        trace_measured,
        window_length=50,
        compare_unit="ms",
        verbose=False,
    ):
        self.trace_ddr = trace_ddr
        self.trace_hbm = trace_hbm
        self.trace_measured = trace_measured

        self.nr_win = None
        self.window_length = None
        self.window_length2 = None
        self.window_length3 = None
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
        if compare_unit == "instrs":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_instruction_bounds_
            self._subset_window_fn_ = Trace.subset_window__instruction
        if compare_unit == "ms":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_timestamp_bounds_
            self._subset_window_fn_ = Trace.subset_window_timestamp
        else:
            err_msg = "Invalid window comparison unit: {}.\n".format(method)
            err_msg += "Expected one of ['accesses', 'instrs', 'ms']"
            raise ValueError(err_msg)
        self.window_length = window_length
        self.compare_unit = compare_unit
        self._set_window_()

    def singlify_phases(self):
        """
        Set all traces in this TraceSet such that there is only the same single
        phase in them.
        """
        self.trace_ddr.singlify_phases()
        self.trace_hbm.singlify_phases()
        if self.trace_measured is not None:
            self.trace_measured.singlify_phases()

    def is_empty(self):
        """
        Return whether there is no sample in this TraceSet.
        """
        return len(self.trace_ddr) == 0 or len(self.trace_hbm) == 0

    def timespan_ddr(self):
        """
        Get the total time in the ddr trace.
        """
        return (
            self.trace_ddr["Timestamp"].iloc[-1] - self.trace_ddr["Timestamp"].iloc[0]
        )

    def timespan_hbm(self):
        """
        Get the total time in the hbm trace.
        """
        return (
            self.trace_hbm["Timestamp"].iloc[-1] - self.trace_hbm["Timestamp"].iloc[0]
        )

    def timespan_measured(self):
        """
        Get the total time in the measured trace.
        If there is no measured trace, 0 is returned.
        """
        if self.trace_measured is None:
            return 0
        return (
            self.trace_measured["Timestamp"].iloc[-1]
            - self.trace_measured["Timestamp"].iloc[0]
        )

    def subset_phases(phases):
        """
        Filter this TraceSet to contain only selected phases.
        """
        self.trace_ddr.subset_phases(phases)
        self.trace_hbm.subset_phases(phases)
        if self.trace_measured is not None:
            self.trace_measured.subset_phases(phases)

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
        if self.trace_measured is not None:
            trace_measured = self._subset_window_fn_(
                self.trace_measured, self.window_length3, win, self.nr_win
            )
        else:
            trace_measured = None
        return TraceSet(
            trace_ddr,
            trace_hbm,
            trace_measured,
            self.window_length,
            self.compare_unit,
            self._verbose_,
        )

    def print_windows_info(self):
        print(
            "{} windows with window_length: {} {}, window_length2: {} {}".format(
                self.nr_wins,
                self.window_length,
                self.compare_unit,
                self.window_length2,
                self.compare_unit,
            )
        )

    def __iter__(self):
        return PhaseIterator(self, self._verbose_)

    def _get_trace_timestamp_bounds_(trace):
        """
        Get bounds of the trace (minimum timestamp, maximum timestamp)
        in the "ms" unit.
        """
        return trace["Timestamp"].iloc[0], trace["Timestamp"].iloc[-1]

    def _get_trace_instruction_bounds_(trace):
        """
        Get bounds of the trace (minimum instruction number,
        maximum instruction number) in the "instruction" unit.
        """
        return trace["Instrs"].iloc[0], trace["Instrs"].iloc[-1]

    def _get_trace_access_bounds_(trace):
        """
        Get bounds of the trace (minimum memory access number,
        maximum memory access number) in the "access" unit.
        """
        return 0, len(trace)

    def _set_window_(self):
        """
        Set number of windows, and window length for each trace according
        to the selected unit.
        """
        t1_s, t1_e = self._get_trace_bounds_fn_(self.trace_ddr)
        t2_s, t2_e = self._get_trace_bounds_fn_(self.trace_hbm)
        self.nr_wins = int((t1_e - t1_s) / self.window_length)
        self.window_length2 = int((t2_e - t2_s) / self.nr_wins)
        if self.trace_measured is not None:
            t3_s, t3_e = self._get_trace_bounds_fn_(self.trace_measured)
            self.window_length3 = int((t3_e - t3_s) / self.nr_wins)


class Phase(TraceSet):
    """
    A Phase is a subset of a TraceSet containing a single phase.
    """

    def __init__(self, phase, trace_set):
        if not isinstance(phase, int):
            raise ValueError(
                "Phase argument must be an integer to subset the " "phase of a TraceSet"
            )

        trace_ddr = trace_set.trace_ddr.get_phases(phase)
        trace_hbm = trace_set.trace_hbm.get_phases(phase)
        if self.trace_measured is not None:
            trace_measured = trace_set.trace_measured.get_phases(phase)
        else:
            trace_measured = None
        compare_unit = trace_set.compare_unit
        window_length = trace_set.compare_unit

        super(Phase, self).__init__(
            trace_ddr, trace_hbm, trace_measured, window_length, compare_unit
        )
        self.phase = phase

        if trace_set._verbose_:
            self.print_info()
            if trace_ddr.is_empty():
                print(
                    "{}: WARNING: no data available in "
                    "phase {}".format(trace_ddr.filename, phase)
                )

    def print_info(self):
        self.print_windows_info()
        self.trace_ddr.print_info(self.phase)
        self.trace_hbm.print_info(self.phase)

        self.trace_measured.print_info(self.phase)


class PhaseIterator:
    """
    TraceSet iterator iterating through Phases of the traces and through windows
    inside phases of the trace. On each iteration, the next window is returned.
    """

    def __init__(self, trace_set):
        # Private
        self._window_iterator_ = None
        self._traces_ = trace_set
        self._phases_ = iter(trace_set.phases())
        self._set_next_phase_()

    def _set_next_phase_(self):
        """
        Move iterator to next phase.
        """
        phase = self._traces_._get_phase_(next(self._phases_), self._traces_._verbose_)
        self._window_iterator_ = WindowIterator(phase)

    def __next__(self):
        """
        Get next window of the traces.
        """
        try:
            return next(self._window_iterator_)
        except StopIteration:
            self._set_next_phase_()
            return next(self)


class WindowIterator:
    """
    TraceSet iterator iterating through windows of the traces.
    On each iteration, the next window is returned.
    """

    def __init__(self, trace_set):
        # Private
        self._traces_ = trace_set
        self._win_ = 0

    def __next__(self):
        """
        Get next window of the traces.
        """
        try:
            window = self._traces_._get_window_(self._win_)
            self._win_ += 1
            return window
        except ValueError:
            raise StopIteration


class Estimator:
    """
    Estimation of execution time for a set of traces and a given set of address
    intervals mapped into the High Bandwidth Memory (hbm).
    """

    def estimate(trace_set, hbm_intervals, hbm_factor=1.0):
        """
        Estimate execution time of the provided trace set and when some ranges
        of addresses are mapped in the hbm memory. The weight of hbm accesses
        can be increased with `hbm_factor` to set the impact of hbm memory
        access over the overall execution time.

        When execution time in hbm is significantly faster than execution in
        ddr (> 3% faster), the estimation is computed with
        `_compute_accurate_estimate_()` function, else it is
        the average between t_ddr and t_hbm (`_compute_fast_estimate_()`).
        """
        # Iterate phases and windows to estimate save time versus actual spent
        # time.
        estimated_time = 0
        measured_time = 0
        for window in trace_set:
            estimated_time += Estimator._estimate_window_(window, hbm_factor)
            measured_time += window.timespan_measured()
        return estimated_time, measured_time

    def _estimate_window_(window, hbm_intervals, hbm_factor):
        """
        This function choses between estimation methods:
        `_compute_accurate_estimate_()` or `_compute_fast_estimate_()`
        """
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()
        if t_ddr > (t__hbm * 1.03):
            return Estimator._compute_accurate_estimate_(
                window, hbm_intervals, hbm_factor
            )
        else:
            return Estimator._compute_fast_estimate_(window)

    def _compute_accurate_estimate_(window, hbm_intervals, hbm_factor):
        """
        This the function computing the actual estimate.
        """
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()

        # Count hbm and ddr accesses
        ddr_accesses = 0
        hbm_accesses = 0
        for vaddr in window.virtual_addresses():
            if hbm_intervals.overlaps(vaddr):
                hbm_accesses += 1
            else:
                ddr_accesses += 1

        # Compute estimate
        total_weighted_accesses = float(ddr_accesses + hbm_accesses * hbm_factor)
        weighted_ratio_of_hbm_access = float(hbm_accesses) / total_weighted_accesses
        max_saved_time = float(t_ddr - t_hbm)
        return t_ddr - (max_saved_time * weighted_ratio_of_hbm_access)

    def _compute_fast_estimate_(self):
        """
        This function computes the average between time in ddr memory
        and time hbm memory. This is accurate enough if both times are nearly
        equal.
        """
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()
        return (t_ddr - t_hbm) / 2.0


trace_dir = "/home/ndenoyelle/Documents/Heterogeneous-Memory-Management/traces"
ddr_input = "{}/DRAM-lulesh2.0-PEBS-countdown-32-PEBS-buffer-4096-pid-198687-tid-198687.dat.feather".format(
    trace_dir
)
hbm_input = "{}/MCDRAM-lulesh2.0-PEBS-countdown-32-PEBS-buffer-4096-pid-196844-tid-196844.dat.feather".format(
    trace_dir
)
measured_trace = None
ddr_trace = Trace(ddr_input, 1400000, verbose=True)
hbm_trace = Trace(hbm_input, 1400000, verbose=True)
traces = TraceSet(ddr_trace, hbm_trace, measured_trace, verbose=True)

if __name__ == "__main__":

    class Parser:
        """
        Class to parse input arguments of the program.
        """

        def parse_addr(addr):
            """
            Parse input address and return the address in `int` type.
            """
            try:
                return int(addr, 0)
            except ValueError:
                raise ValueError(
                    "Invalid address format: {}.\n" "Expected: '<hex>'".format(addr)
                )

        def parse_interval(interval):
            """
            Parse input interval and return a tuple of the smallest and highest
            addresses with `int` type.
            """
            intrvl = interval.split(":")
            if len(intrvl) != 2:
                raise ValueError(
                    "Invalid interval format {}.\n"
                    "Expected '<hex>:<hex>'".format(interval)
                )
            low = Parse.parse_addr(intrvl[0])
            high = Parse.parse_addr(intrvl[1])
            return (low, high) if low < high else (high, low)

        def parse_intervals(intervals):
            """
            Parse a list of intervals return a merged IntervalTree() of the
            intervals.
            """
            intrvls = IntervalTree()
            for i in intervals:
                low, high = Parse.parse_interval(i)
                intrvls[low:high] = None
            intrvls.merge_overlaps()
            return intrvls

        def parse_name_intervals(name):
            """
            Parse hbm intervals contained in the filename of the traces
            containing measured experiment with arbitrary allocation of data
            in hbm and ddr memories.
            """
            inputs = os.path.basename(name).split("-")
            inputs = [
                "{}:{}".format(inputs[i + 1], inputs[i + 2])
                for i in inputs[::3]
                if i == "HBM"
            ]
            return Parse.parse_intervals(inputs)

        def parse_phases(phases):
            """
            Format input phases in a `int` list.
            """
            return [int(i) for i in phases]

    def print_intervals(intervals, prefix=""):
        """
        Pretty print address intervals.
        """
        intervals = (
            "{}{}:{}".format(prefix, "0x%x" % i.begin, "0x%x" % i.end)
            for i in sorted(intervals)
        )
        print("\n".join(intervals))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddr-input",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in ddr memory."
        ),
    )
    parser.add_argument(
        "--hbm-input",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in hbm memory."
        ),
    )
    parser.add_argument(
        "--measured-input",
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution to be compared with estimator."
        ),
    )
    parser.add_argument(
        "--cpu-cycles-per-ms",
        default=1400000,
        type=int,
        help="CPU cycles per millisecond (default: 1,400,000 for KNL).",
    )
    parser.add_argument(
        "--compare-window-len", default=50, type=int, help="Comparison window length."
    )
    parser.add_argument(
        "--compare-window-unit",
        default="ms",
        type=str,
        help="Comparison window length unit (accesses, ms or instrs).",
    )
    parser.add_argument(
        "--hbm-factor", default=1.0, type=float, help="HBM weight factor."
    )
    parser.add_argument(
        "--hbm-ranges",
        action="append",
        type=str,
        help=(
            "Memory ranges that are placed in high-bandwidth memory "
            "(e.g., 0x2aaaf9144000-0x2aaafabd0000)"
        ),
    )
    parser.add_argument(
        "-p",
        "--phases",
        action="append",
        type=int,
        help="Subset traces to only contain theses phases.",
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    # Load traces and bundle them together.
    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms, args.verbose)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms, args.verbose)
    if args.measured_input is not None:
        measured_trace = Trace(
            args.measured_input, args.cpu_cycles_per_ms, args.verbose
        )
    else:
        measured_trace = None
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
        measured_trace,
        args.compare_window_len,
        args.compare_window_unit,
        args.verbose,
    )
    # Filter phases if needed.
    if args.phases is not None:
        phases = [int(i) for i in args.phases]
        traces.subset_phases(phases)

    # Compute hbm intervals.
    hbm_intervals = IntervalTree()
    if args.measured_input is not None:
        hbm_intervals.update(Parse.parse_name_intervals(args.measured_input))
    if args.hbm_ranges is not None:
        for hbm_range in args.hbm_ranges:
            hbm_intervals.update(Parse.parse_intervals(args.hbm))
    if args.verbose:
        Memory.print_intervals(hbm_intervals, "HBM range: ")

    # Compute estimation for the traces and hbm_intervals.
    estimated_time, measured_time = Estimator(hbm_intervals).estimate(
        traces, args.hbm_factor
    )
    print("Time measure: {}".format(measured_time))
    print("Time estimated: {}".format(estimated_time))
