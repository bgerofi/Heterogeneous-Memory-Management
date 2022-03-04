# /usr/bin/python
import numpy as np
from memai.traces import WindowIterator


class Estimator:
    """
    Estimation of execution time for a set of traces and a given set of address
    intervals mapped into the High Bandwidth Memory (hbm).
    """

    def __init__(
        self,
        application_traces,
        page_size=13,
        compare_unit="ms",
        window_length=50,
        verbose=False,
    ):
        # Time spent on empty window
        self._empty_time = 0.0
        # Time spent on windows where hbm and ddr had about the same time.
        self._fast_time = 0.0
        self._verbose_ = verbose
        self._hbm_time = []
        self._ddr_time = []
        self._pages = []

        page_mask = Estimator.page_mask(page_size)
        window_iter = WindowIterator(application_traces, compare_unit, window_length)

        for window in window_iter:
            if verbose:
                print(window)

            if window.is_empty():
                self._empty_time += window.t_ddr
                continue

            if False:  # hbm_time * 1.03 >= ddr_time:
                fast_time = Estimator.estimate_fast(window.t_ddr, window.t_hbm)
                self._fast_time += fast_time
            else:
                addr, count = np.unique(
                    window.addresses & page_mask,
                    return_counts=True,
                )
                pages = list(zip(addr, count))
                self._ddr_time.append(window.t_ddr)
                self._hbm_time.append(window.t_hbm)
                self._pages.append(pages)

        self._cumulated_ddr_time = (
            sum(self._ddr_time) + self._fast_time + self._empty_time
        )
        self._cumulated_hbm_time = (
            sum(self._hbm_time) + self._fast_time + self._empty_time
        )

    @staticmethod
    def page_mask(page_size):
        return ~((1 << int(page_size)) - 1)

    def print_estimate_info(self, estimated_time):
        print(
            "Estimation time breakdown:\n"
            "\t{:.2f} (s) on empty windows\n"
            "\t{:.2f} (s) on similar windows\n"
            "\t{:.2f} (s) on estimated windows\n".format(
                self._empty_time / 1000.0,
                self._fast_time / 1000.0,
                estimated_time / 1000.0,
            )
        )

    def estimate(self, hbm_intervals, hbm_factor=1.0):
        """
        Estimate execution time of the estimator traces with a specific mapping
        of pages in the hbm memory. The weight of hbm accesses
        can be increased with `hbm_factor` to set the impact of hbm memory
        access over the overall execution time.
        """
        estimated_time = 0.0

        for (t_ddr, t_hbm, pages) in zip(self._ddr_time, self._hbm_time, self._pages):
            estimated_time += self._estimate_accurate_(
                pages, hbm_intervals, hbm_factor, t_ddr, t_hbm
            )

        if self._verbose_:
            self.print_estimate_info(estimated_time)
        return estimated_time + self._empty_time + self._fast_time

    @staticmethod
    def estimate_fast(t_ddr, t_hbm):
        return (t_ddr + t_hbm) / 2.0

    def _estimate_accurate_(self, pages, hbm_intervals, hbm_factor, t_ddr, t_hbm):
        if len(pages) == 0:
            return 0.0

        ddr_accesses = 0
        hbm_accesses = 0
        for addr, n in pages:
            if hbm_intervals.overlaps_point(addr):
                hbm_accesses += n
            else:
                ddr_accesses += n

        # total_accesses = ddr_accesses + hbm_accesses
        # return (t_ddr * ddr_accesses + t_hbm * hbm_accesses) / total_accesses

        weighted_hbm_accesses = float(hbm_accesses) * hbm_factor
        hbm_saved_time = t_ddr - t_hbm
        hbm_saving_factor = float(weighted_hbm_accesses) / float(
            ddr_accesses + weighted_hbm_accesses
        )
        return t_ddr - (hbm_saved_time * hbm_saving_factor)


if __name__ == "__main__":
    import time
    import re
    from memai import Trace, TraceSet
    from intervaltree import Interval, IntervalTree
    import argparse

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
            low = Parser.parse_addr(intrvl[0])
            high = Parser.parse_addr(intrvl[1])
            return (low, high) if low < high else (high, low)

        def parse_intervals(intervals):
            """
            Parse a list of intervals return a merged IntervalTree() of the
            intervals.
            """
            intrvls = IntervalTree()
            for i in intervals:
                low, high = Parser.parse_interval(i)
                intrvls[low:high] = None
            intrvls.merge_overlaps()
            return intrvls

        def parse_name_intervals(name):
            """
            Parse hbm intervals contained in the filename of the traces
            containing measured experiment with arbitrary allocation of data
            in hbm and ddr memories.
            """
            regex = re.compile("HBM-(?P<begin>0x[a-fA-F0-9]+)-(?P<end>0x[a-fA-F0-9]+)-")
            matches = regex.findall(name)
            if matches is None or len(matches) == 0:
                print(
                    "Warning measure file name does not contain HBM ranges. "
                    "Must contain: HBM-<hex>-<hex>".format(name)
                )
                return IntervalTree()
            intervals = [Interval(int(m[0], 0), int(m[1], 0)) for m in matches]
            intervals = IntervalTree(intervals)
            intervals.merge_overlaps()
            return intervals

        def parse_phases(phases):
            """
            Format input phases in a `int` list.
            """
            return [int(i) for i in phases]

    def time_estimation(estimator, hbm_intevals, hbm_factor):
        t_start = time.time()
        t_e = estimator.estimate(hbm_intevals, hbm_factor)
        t_end = time.time()
        return (t_end - t_start), t_e

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddr-input",
        metavar="<file>",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in ddr memory."
        ),
    )
    parser.add_argument(
        "--hbm-input",
        metavar="<file>",
        required=True,
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution in hbm memory."
        ),
    )
    parser.add_argument(
        "--measured-input",
        metavar="<file>",
        type=str,
        help=(
            "Feather format pandas memory access trace input file of "
            "execution to be compared with estimator."
        ),
    )
    parser.add_argument(
        "--cpu-cycles-per-ms",
        metavar="<int>",
        default=1400000,
        type=int,
        help="CPU cycles per millisecond (default: 1,400,000 for KNL).",
    )
    parser.add_argument(
        "--window-len",
        metavar="<int>",
        default=None,
        type=int,
        help="Comparison window length.",
    )
    parser.add_argument(
        "--compare-unit",
        default="ms",
        choices=["accesses", "ms", "instrs"],
        type=str,
        help="Comparison length unit.",
    )
    parser.add_argument(
        "--hbm-factor",
        metavar="<float>",
        default=1.0,
        type=float,
        help="HBM weight factor.",
    )
    parser.add_argument(
        "--hbm-ranges",
        metavar="<hex-hex>",
        action="append",
        type=str,
        help=(
            "Memory ranges that are placed in high-bandwidth memory "
            "(e.g., 0x2aaaf9144000:0x2aaafabd0000)"
        ),
    )
    parser.add_argument(
        "-p",
        "--filter-phases",
        metavar="<int>",
        action="append",
        type=int,
        help="Subset traces to only contain these phases.",
    )
    parser.add_argument(
        "--singlify-phases",
        default=False,
        action="store_true",
        help="Mark all the phases in the trace as phase 0. This option is "
        "treated after --filter-phases option.",
    )
    parser.add_argument(
        "--page-size",
        metavar="<int>",
        default=13,
        type=int,
        help="The size of a page in number of bits.",
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    # Load traces and bundle them together.
    ddr_trace = Trace(args.ddr_input, args.cpu_cycles_per_ms, args.verbose)
    hbm_trace = Trace(args.hbm_input, args.cpu_cycles_per_ms, args.verbose)
    traces = TraceSet(
        ddr_trace,
        hbm_trace,
    )
    # Filter phases if needed.
    if args.filter_phases is not None:
        phases = [int(i) for i in args.phases]
        traces.subset_phases(phases)
    # Make a single phase if needed.
    if args.singlify_phases:
        traces.singlify_phases()

    # Compute hbm intervals.
    hbm_intervals = IntervalTree()
    if args.measured_input is not None:
        hbm_intervals = Parser.parse_name_intervals(args.measured_input)
    if args.hbm_ranges is not None:
        hbm_intervals = IntervalTree()
        for hbm_range in args.hbm_ranges:
            hbm_intervals.update(Parser.parse_intervals(args.hbm_ranges))
    if len(hbm_intervals) == 0:
        raise ValueError("No HBM intervals provided.")

    # Compute estimation for the traces and hbm_intervals.
    estimator = Estimator(
        traces, args.page_size, args.compare_unit, args.window_len, args.verbose
    )
    runtime, estimated_time = time_estimation(estimator, hbm_intervals, args.hbm_factor)
    hbm_time = hbm_trace.timespan()
    ddr_time = ddr_trace.timespan()
    measured_time = (
        Trace(args.measured_input, args.cpu_cycles_per_ms, args.verbose).timespan()
        if args.measured_input is not None
        else 0
    )

    error = float(abs(estimated_time - measured_time)) / float(abs(ddr_time - hbm_time))
    print("Time DDR: {:.2f} (s)".format(ddr_time / 1000.0))
    print("Time HBM: {:.2f} (s)".format(hbm_time / 1000.0))
    if args.measured_input is not None:
        print("Time Measured: {:.2f} (s)".format(measured_time / 1000.0))
    print("Time Estimated: {:.2f} (s)".format(estimated_time / 1000.0))
    print(
        "Cumulated DDR Windows Time: {:.2f} (s)".format(
            estimator._cumulated_ddr_time / 1000.0
        )
    )
    print(
        "Cumulated HBM Windows Time: {:.2f} (s)".format(
            estimator._cumulated_hbm_time / 1000.0
        )
    )
    if args.measured_input is not None:
        print("Estimation Relative Error: {:.2f}%".format(100.0 * error))
    print("Estimator Runtime: {:.8f} (s)".format(runtime))
    if args.measured_input is not None:
        print("Estimator Speedup: {:.0f}".format(measured_time / (1000.0 * runtime)))
