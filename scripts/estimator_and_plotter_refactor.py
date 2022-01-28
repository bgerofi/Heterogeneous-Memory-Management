#/usr/bin/python
import os
import pandas as pd
import numpy as np
import argparse
from intervaltree import Interval, IntervalTree
import importlib

class Memory:
    # The size of page: 8192 (Bits).
    PAGE_SIZE=1<<13
    # Mask to get the page low_vaddr.
    PAGE_MASK=~(PAGE_SIZE-1)
    hex_range=np.vectorize(hex)

    def page_addr(addr):
        return addr & Memory.PAGE_MASK

    def print_intervals(intervals, prefix=""):
        print('\n'.join([ '{}{}:{}'.format(prefix, '0x%x'%i.begin, '0x%x'%i.end) for i in sorted(intervals) ]))

class Trace:
    '''
    Class representing a single trace file.
    The file contains one or several phases and can subseted to one of these.
    It also be split into windows and subseted to one of these. The window
    properties are defined by the user when subsetting, and the metric used for
    for subsetting is also to be chosen.
    '''
    def __init__(self, filename, cpu_cycles_per_ms,
                 compare_window, compare_method,
                 verbose=True):
        if verbose:
            print("Loading from {}...".format(filename))

        # Private
        self._data_ = pd.read_feather(filename)
        self._data_["Timestamp"] = data["Timestamp"].div(cpu_cycles_per_ms)
        # Public
        self.filename = filename

    def is_empty(self):
        return len(self_data_) == 0
        
    def singlify_phases(self):
        self._data_.assign(Phase=0)

    def phases(self):
        return np.unique(self._data_["Phase"])

    def print_info(self):
        print("Loaded {} accesses ({} phases) between {} and {} msecs".format(
            len(self._data_), len(self.phases()),
            self._data_["Timestamp"].iloc[0], self._data_["Timestamp"].iloc[-1]))
        
    def _subset_(self, cond):
        t = Trace.__new__(Trace)
        t.filename = self.filename
        t.data = self._data_[cond]
        return t

    def get_phases(self, phase):
        if isinstance(phase, int):
            phase = [phase]
        if not isinstance(phase, list):
            raise TypeError("Provided phase must be either a number or a list of numbers.")

        available_phases = self.phases()
        if any((p not in available_phases for p in phases)):
            raise ValueError("Invalid phase {}. Possible values are: {}"\
                             .format(phase, available_phases))
        return self._subset_(self._data_.Phase in phase)

    def subset_window_timestamp(self, compare_window, win, nr_wins):
        return self._subset_((self._data_["Timestamp"] >= (compare_window * win)) & \
                           (self._data_["Timestamp"] < (compare_window * (win + 1))))

    def subset_window_instruction(self, compare_window, win, nr_wins):
        if win == nr_wins - 1:
            return self._subset_(self._data_["Instrs"] >= (compare_window * win))
        else:
            return self._subset_((self._data_["Instrs"] >= (compare_window * win)) & \
                               (self._data_["Instrs"] < (compare_window * (win + 1))))

    def subset_window_access(self, compare_window, win, nr_wins):
        if win == nr_wins - 1:
            return self._subset_((compare_window * win):)
        else:
            return self._subset_((compare_window * win):(compare_window * (win + 1)))

    def print_phase_info(self, phase):
        print("{}: {} accesses in phase {}, "
              "time window between {} and {} ({}) msecs, "
              "retired instructions: {}, mem/instr: {}"\
              .format(self._data_.filename, len(self._data_), phase,
                      self._data_["Timestamp"].iloc[0],
                      self._data_["Timestamp"].iloc[-1],
                      self._data_["Timestamp"].iloc[-1] - self._data_["Timestamp"].iloc[0],
                      self._data_["Instrs"].iloc[-1],
                      len(self._data_) * 8 / trace["Instrs"].iloc[-1]))

    def __len__(self):
        return self._data_.__len__()

    def __get_item__(self, item):
        return self._data_.__get_item__(item)

class TraceSet:
    def __init__(self, trace_ddr, trace_hbm, trace_measured,
                compare_window, compare_method):
        self.trace_ddr = trace_ddr
        self.trace_hbm = trace_hbm
        self.trace_measured = trace_measured

        self.nr_win = None
        self.compare_window = None        
        self.compare_window2 = None
        self.compare_window3 = None
        self.set_window_properties(compare_window, compare_method)
        
    def set_window_properties(self, compare_window, compare_method):
        if compare_method == "accesses":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_access_bounds_
            self._subset_window_fn_ = Trace.subset_window_access
        if compare_method == "instrs":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_instruction_bounds_
            self._subset_window_fn_ = Trace.subset_window__instruction
        if compare_method == "ms":
            self._get_trace_bounds_fn_ = TraceSet._get_trace_timestamp_bounds_
            self._subset_window_fn_ = Trace.subset_window_timestamp
        else:
            err_msg = "Invalid window comparison unit: {}.\n".format(method)
            err_msg += "Expected one of ['accesses', 'instrs', 'ms']"
            raise ValueError(err_msg)
        self.compare_window = compare_window
        self.compare_method = compare_method
        self._set_window_()

    def singlify_phases(self):
        self.trace_ddr.singlify_phases()
        self.trace_hbm.singlify_phases()
        if self.trace_measured is not None:
            self.trace_measured.singlify_phases()

    def is_empty(self):
        return len(self.trace_ddr) == 0 || len(self.trace_hbm) == 0

    def time_start(self):
        return self.trace_ddr["Timestamp"].iloc[0]

    def time_end(self):
        return self.trace_ddr["Timestamp"].iloc[-1]
    
    def timespan_ddr(self):
        return self.time_end() - self.time_start()

    def timespan_hbm(self):
        return self.trace_hbm["Timestamp"].iloc[-1] - self.trace_hbm["Timestamp"].iloc[0]
    
    def time_measured(self):
        if self.trace_measured is None:
            return 0
        else:
            return self.trace_measured["Timestamp"].iloc[-1] - self.trace_measured["Timestamp"].iloc[0]
    
    def subset_phases(phases):
        self.trace_ddr.subset_phases(phases)
        self.trace_hbm.subset_phases(phases)
        if self.trace_measured is not None:
            self.trace_measured.subset_phases(phases)
            
    def get_phase(self, phase):
        return Phase(phase, self)

    def get_window(self, win):
        if win >= self.nr_win:
            raise ValueError("Window {} out of bounds ({}).".format(win, self.nr_win))
        trace_ddr = self._subset_window_fn_(self.trace_ddr, self.compare_window,
                                         win, self.nr_win)
        trace_hbm = self._subset_window_fn_(self.trace_hbm, self.compare_window2,
                                         win, self.nr_win)
        trace_measured = None
        if self.trace_measured is not None:
            trace_measured = self._subset_window_fn_(self.trace_measured, self.compare_window3,
                                             win, self.nr_win)
        return TraceSet(trace_ddr, trace_hbm, trace_measured,
                        self.compare_window, self.compare_method)

    def print_windows_info(self):
        print("{} windows with compare_window: {} {}, compare_window2: {} {}"\
              .format(self.nr_wins, self.compare_window, self.compare_method,
                      self.compare_window2, self.compare_method))

    def iter_windows(self):
        return WindowIterator(self)

    def iter_phases(self):
        return PhaseIterator(self)

    def _get_trace_timestamp_bounds_(trace):
        return trace["Timestamp"].iloc[0], trace["Timestamp"].iloc[-1]

    def _get_trace_instruction_bounds_(trace):
        return trace["Instrs"].iloc[0], trace["Instrs"].iloc[-1]

    def _get_trace_access_bounds_(trace):
        return 0, len(trace)

    def _set_window_(self):
        t1_s, t1_e = self._get_trace_bounds_fn_(self.trace_ddr)
        t2_s, t2_e = self._get_trace_bounds_fn_(self.trace_hbm)        
        self.nr_wins = int((t1_e - t1_s) / self.compare_window)
        self.compare_window2 = int((t2_e - t2_s) / self.nr_wins)
        if self.trace_measured is not None:
            t3_s, t3_e = self._get_trace_bounds_fn_(self.trace_measured)
            self.compare_window3 = int((t3_e - t3_s) / self.nr_wins)

class Phase(TraceSet):
    def __init__(self, phase, trace_set):
        if not isinstance(phase, int):
            raise ValueError("Phase argument must be an integer to subset the "
                             "phase of a TraceSet")
        
        trace_ddr = trace_set.trace_ddr.get_phases(phase)
        trace_hbm = trace_set.trace_hbm.get_phases(phase)
        trace_measured = None
        if trace_set.trace_measured is not None:
            trace_measured = trace_set.trace_measured.get_phases(phase)
        if trace_measured is not None and trace_measured.is_empty():
            trace_measured = None
        compare_method = trace_set.compare_method
        compare_window = trace_set.compare_method
        super(Phase, self).__init__(trace_ddr, trace_hbm, trace_measured,
                                    compare_window, compare_method)
        if trace_ddr.is_empty():
            print("{}: WARNING: no data available in phase {}".format(trace_ddr.filename, phase))
        self.phase = phase

    def print_info(self):
        self.print_windows_info()
        self.trace_ddr.print_info(self.phase)
        self.trace_hbm.print_info(self.phase)
        if trace_measured is not None:
            self.trace_measured.print_info(self.phase)

class PhaseIterator:
    def __init__(self. trace_set):
        # Public
        self.time_total = 0
        self.time_measured = 0
        self.window_iterator = None

        # Private
        self._prev_phase_last_ts_ = -1
        self._traces_ = trace_set
        self._phases_ = iter(trace_set.phases())
        self._set_next_phase_()

    def _set_next_phase_(self):
        phase = self._traces_.get_phase(next(self._phases_))
        if phase.is_empty():
            if self._prev_phase_last_ts_ > -1:
                self.time_total += phase.time_start() - self._prev_phase_last_ts_
                self._prev_phase_last_ts_ = -1
            else:
                self._prev_phase_last_ts_ = phase.time_start()
        else:
            self.time_total += phase.timespan_ddr()
            self._prev_phase_last_ts_ = phase.time_end()
        
        self.time_measured += phase.time_measured()
        self.window_iterator = WindowIterator(phase)

    def __next__(self):
        try:
            return next(self.window_iterator)
        except StopIteration:
            self._set_next_phase_()
            return next(self)
        
class WindowIterator:
    def __init__(self, trace_set):
        # Private
        self._traces_ = trace_set
        self._win_ = 0
        self._prev_win_last_ts_ = -1
    
    def __next__(self):
        try:
            window = self._traces_.get_window(self._win_)
        except ValueError:
            raise StopIteration
        self._win_ += 1
        return window

class Estimator:
    def __init__(self, hbm_intervals, hbm_factor=1.0):
        self.hbm_intervals = hbm_intervals
        self.hbm_factor = hbm_factor

    def compute_slow_estimate(self, window):
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()
        nr_accesses = 0
        nr_hbm_accesses = 0
        for vaddr in window.trace_ddr["Vaddr"].values:
            if self.hbm_intervals.overlaps(vaddr):
                nr_hbm_accesses += 1
            else:
                nr_accesses += 1
        nr_accesses += nr_hbm_accesses * self.hbm_factor
        return t_ddr - (float(t_ddr - t_hbm) * (float(nr_hbm_accesses) / nr_accesses))

    def compute_slow_estimate(self):
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()
        return (t_ddr - t_hbm) / 2.0

    def estimate(self, window):
        t_ddr = window.timespan_ddr()
        t_hbm = window.timespan_hbm()
        if t_ddr > (t__hbm * 1.03):
            return self.compute_slow_estimate(window)
        else:
            return self.compute_fast_estimate(window)
        
if __name__ == "__main__":
    class Parser:
        def parse_addr(addr):
            try:
                return int(addr, 0)
            except ValueError:
                raise ValueError("Invalid address format: {}.\n"
                                 "Expected: '<hex>'".format(addr))
            
        def parse_interval(interval):
            intrvl = interval.split(':')
            if len(intrvl) != 2:
                raise ValueError("Invalid interval format {}.\n"
                                 "Expected '<hex>:<hex>'".format(interval))
            low = Parse.parse_addr(intrvl[0])
            high = Parse.parse_addr(intrvl[1])
            return (low, high) if low < high else (high, low)

        def parse_intervals(intervals):
            intrvls = IntervalTree()
            for i in intervals:
                low, high = Parse.parse_interval(i)
                intrvls[low:high] = None
            intrvls.merge_overlaps()
            return intrvls

        def parse_name_intervals(name):
            inputs = os.path.basename(name).split("-")
            inputs = [ '{}:{}'.format(inputs[i+1], inputs[i+2]) \
                       for i in inputs[::3] if i == "HBM" ]
            return Parse.parse_intervals(inputs)

        def parse_phases(phases):
            return [ int(i) for i in phases ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
        action='append',
        help=("Feather format pandas memory access trace input file(s)."))
    parser.add_argument('--cpu-cycles-per-ms', default=1400000, type=int,
        help='CPU cycles per millisecond (default: 1,400,000 for KNL).')
    parser.add_argument('--phase', action='append', type=int,
        help='Specify application phase.')
    parser.add_argument('--compare-window-len', default=50, type=int,
        help='Comparison window length.')
    parser.add_argument('--compare-window-unit', default="ms", type=str
        help='Comparison window length unit (accesses, ms or instrs).')
    parser.add_argument('--hbm-factor', default=1.0, type=float,
        help='HBM weight factor.')
    parser.add_argument("--hbm", action='append', type=str,
        help=("Memory ranges that are placed in high-bandwidth memory (e.g., 0x2aaaf9144000-0x2aaafabd0000)"))
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    # Compute hbm intervals.
    hbm_intervals = IntervalTree()
    if args.hbm is not None:
        hbm_intervals.update(Parse.parse_intervals(args.hbm))
    if data3 is not None:
        hbm_intervals.update(Parse.parse_name_intervals(data3.filename))
    if not args.verbose:
        Memory.print_intervals(hbm_intervals, 'HBM range: ')
        
    # Load traces
    data = Trace(args.input[0], args.cpu_cycles_per_ms)
    if not args.csv:
        data.print_info()
    data2 = Trace(args.input[1], args.cpu_cycles_per_ms) if len(args.input) > 1 else None
    if data2 is not None and not args.csv:
        data2.print_info()
    data3 = Trace(args.input[2], args.cpu_cycles_per_ms) if len(args.input) > 2 else None
    if data3 is not None and not args.csv:
        data3.print_info()

    # Bundle traces together.
    traces = TraceSet(data, data2, data3,
                      args.compare_window_len, args.compare_window_unit)
    if args.phase is not None:
        if len(args.phase) == 1 and args.phase[0] == -1:
            traces.singlify_phases()
        else:
            traces.subset_phases(Parser.parse_phases(args.phase))
        if traces.is_empty():
            raise ValueError("No data available in selected phases")

    
    
    for window in traces.iter_phases(args.detect_intervals):
        pages = {}
        pages2 = {}
        inter = {}
        interval_pairs = IntervalTree()
        PAGE_SIZE = 4096
        PAGE_SHIFT = 12
        nr_hbm_accesses = 0
        nr_accesses = 0

        def is_hbm(nr_hbm_accesses, vaddr):
            if hbm_intervals.overlaps(vaddr):
                nr_hbm_accesses += 1
