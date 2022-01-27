#/usr/bin/python
import sys
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans
from kneed import KneeLocator
from intervaltree import Interval, IntervalTree
import importlib
import time

class Memory:
    # The size of page: 8192 (Bits).
    PAGE_SIZE=1<<13
    # Mask to get the page low_vaddr.
    PAGE_MASK=~(PAGE_SIZE-1)

    def _page_low_addr_(vaddr):
        return Memory.PAGE_MASK & vaddr
    def _interval_low_addr_(vaddr, interval_distance):
        return Memory._page_low_addr_(vaddr) - interval_distance * Memory.PAGE_SIZE
    def _interval_high_addr_(vaddr, interval_distance):
        return Memory._page_low_addr_(vaddr + interval_distance * Memory.PAGE_SIZE)

    def detect_intervals(address_list, interval_distance, verbose=False):
        intervals = IntervalTree()
        if verbose:
            print("Detecting address intervals...")
        for vaddr in address_list:
            # Align to page size
            low_vaddr = Memory.interval_low_addr(vaddr, interval_distance)
            high_vaddr = Memory.interval_high_addr(vaddr, interval_distance)
            intervals[low_vaddr:high_vaddr] = 1
            intervals.merge_overlaps(data_reducer = lambda a,b: a+b)
        return intervals

    def print_intervals(intervals, prefix=""):
        print('\n'.join([ '{}{}:{}'.format(prefix, '0x%x'%i.begin, '0x%x'%i.end) for i in sorted(intervals) ]))

class Parser:
    def parse_addr(addr):
        try:
            return int(addr, 0)
        except ValueError:
            raise ValueError("Invalid address format: {}.\nExpected: '<hex>'".format(addr))

    def parse_interval(interval):
        intrvl = interval.split(':')
        if len(intrvl) != 2:
            raise ValueError("Invalid interval format {}.\nExpected '<hex>:<hex>'".format(interval))
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

class Trace:
    def __init__(self, filename, cpu_cycles_per_ms, verbose=True):
        if verbose:
            print("Loading from {}...".format(filename))
        self.data = pd.read_feather(filename)
        self.data["Timestamp"] = data["Timestamp"].div(cpu_cycles_per_ms)
        self.filename = filename

    def singlify_phases(self):
        self.data.assign(Phase=0)

    def print_info(self):
        print("Loaded {} accesses ({} phases) between {} and {} msecs".format(
            len(trace), trace["Phase"].iloc[-1], trace["Timestamp"].iloc[0],
            trace["Timestamp"].iloc[-1]))

def main():
    vhex = np.vectorize(hex)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
        action='append',
        help=("Feather format pandas memory access trace input file(s)."))
    parser.add_argument("-o", "--output", type=str, required=False,
        default="out.pdf",
        help=("Output file name."))
    parser.add_argument('-S', '--start-timestamp', required=False,
        type=int, help='Start timestamp (in msec).')
    parser.add_argument('-E', '--end-timestamp', required=False,
        type=int, help='End timestamp (in msec).')
    parser.add_argument('--cpu-cycles-per-ms', required=False, default=1400000,
        type=int, help='CPU cycles per millisecond (default: 1,400,000 for KNL).')

    parser.add_argument('-L', '--low-vaddr', required=False,
        type=Parser.parse_addr,
        help='Low virtual address.')
    parser.add_argument('-H', '--high-vaddr', required=False,
        type=Parser.parse_addr,
        help='High virtual address.')

    parser.add_argument('--phases', default=False, action='store_true',
        help='Print a summary of each application phase.')
    parser.add_argument('--phase', required=False, type=int, action='append',
        help='Specify application phase.')
    parser.add_argument('--phase-window', required=False, default=None,
        type=float, help='Phase iteration window length (msecs).')
    parser.add_argument('--phase-steps', required=False, default=None,
        type=int, help='Phase iteration steps number.')
    parser.add_argument('--compare-phase', required=False, action='store_true',
        help='Compare a phase between two logs.')

    parser.add_argument('--range', action='append', required=False,
        help='Virtual range(s) to process (format: start_addr-end_addr).')

    parser.add_argument('--detect-intervals', default=False, action='store_true')
    parser.add_argument('--interval-distance', required=False, default="512",
        type=int, help='Max number of pages between accesses that belong to the same interval.')

    parser.add_argument('--compare', default=False, action='store_true', help="Compare two traces.")
    parser.add_argument('--compare-window-len', required=False, default=50,
        type=int, help='Comparison window length.')
    parser.add_argument('--compare-window-unit', required=False, default="ms",
        help='Comparison window length unit (accesses, ms or instrs).')

    parser.add_argument('--detect-clusters', default=False, action='store_true')
    parser.add_argument('--nr-clusters', required=False,
        type=int, help='Number of clusters to use.')
    parser.add_argument('--cluster-id', required=False,
        type=int, help='Address cluster ID.')
    parser.add_argument('--cluster-width', required=False,
        type=int, help='Cluster width in number of pages (4kB sized).')

    parser.add_argument('--min-accesses', required=False, default=100,
        type=int, help='Minimum number of accesses in a cluster/interval to plot.')

    parser.add_argument('--hbm-factor', required=False, default=1.0,
        type=float, help='HBM weight factor.')
    parser.add_argument("--hbm", type=str, required=False,
        action='append',
        help=("Memory ranges that are placed in high-bandwidth memory (e.g., 0x2aaaf9144000-0x2aaafabd0000)"))
    parser.add_argument('--estimate', default=False, action='store_true',
        help=("Estimate runtime based on two traces and the specified HBM ranges."))
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--csv', default=False, action='store_true',
        help="Output results in CSV format")

    args = parser.parse_args()

    if args.detect_intervals and args.interval_distance is None:
        print("error: you must specify --interval-distance when detecting intervals")
        sys.exit(-1)

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

    # Compute hbm intervals.
    hbm_intervals = IntervalTree()
    if args.hbm is not None:
        hbm_intervals.update(Parse.parse_intervals(args.hbm))
    if data3 is not None:
        hbm_intervals.update(Parse.parse_name_intervals(data3.filename))
    if not args.csv:
        Memory.print_intervals(hbm_intervals, 'HBM range: ')
        
    if args.compare_phase:
        sys.exit(0)

    '''
    # Runtime estimator
    if args.estimate and not args.compare:
        if data2 is None:
            print("error: you must specify two input files for estimation")
            sys.exit(-1)

        if args.phase is not None:
            data = data[data["Phase"] == int(args.phase)]
            data2 = data2[data2["Phase"] == int(args.phase)]
            if not args.csv:
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
                    args.input[0], len(data), args.phase, data["Timestamp"].iloc[0], data["Timestamp"].iloc[-1],
                        data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0], data["Instrs"].iloc[-1]))
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
                    args.input[1], len(data2), args.phase, data2["Timestamp"].iloc[0], data2["Timestamp"].iloc[-1],
                        data2["Timestamp"].iloc[-1] - data2["Timestamp"].iloc[0], data2["Instrs"].iloc[-1]))


        prev_phase = -1
        phase = -1

        nr_hbm_accesses = 0
        nr_accesses = 0

        t_all = 0

        for i in range(len(data)):
            if phase == -1:
                phase = data["Phase"].iloc[i]
                prev_phase = data["Phase"].iloc[i]

            phase = data["Phase"].iloc[i]

            # New phase or end of data?
            if prev_phase != phase or i == (len(data) - 1):
                data_phase = data[data["Phase"] == prev_phase]
                data2_phase = data2[data2["Phase"] == prev_phase]

                if len(data_phase) > 0 and len(data2_phase) > 0:
                    t2 = data2_phase["Timestamp"].iloc[-1] - data2_phase["Timestamp"].iloc[0]
                    t = data_phase["Timestamp"].iloc[-1] - data_phase["Timestamp"].iloc[0]

                    if not args.csv:
                        print("phase: {}: t: {}, t2: {}, nr_hbm_accesses: {}, nr_accesses: {}".format(
                            prev_phase, t, t2, nr_hbm_accesses, nr_accesses))
                    if t2 < t:
                        t = int(t - (float(t - t2) * (float(nr_hbm_accesses) / nr_accesses)))

                    if not args.csv:
                        print("phase: {}: estimated runtime: {} msecs (full HBM: {} msecs)".format(prev_phase, t, t2))
                    t_all += t

                prev_phase = phase
                nr_hbm_accesses = 0
                nr_accesses = 0


            if hbm_intervals.overlaps(data["Vaddr"].iloc[i]):
                nr_hbm_accesses += 1

            nr_accesses += 1

        if not args.csv:
            print("Estimated overall runtime: {} msecs".format(t_all))

        sys.exit(0)
    '''

    # Traces comparison
    if args.compare:
        # Estimated overall runtime (if estimating)
        t_all = 0
        t_measured_all = 0
        PAGE_SIZE = 32768
        PAGE_SIZE = 65536
        csv_estimate_postfix = " (Phasemarked)"
        if len(args.input) < 2:
            print("error: you must specify two traces when comparing..")
            sys.exit(-1)

        if data2 is None:
            print("error: you must specify two input files for comparison")
            sys.exit(-1)

        # Convert traces to a single phase if args.phase == [-1]:
        if args.phase and len(args.phase) == 1 and args.phase[0] == -1:
            t_s = time.time()
            '''
            cumm_instr = 0
            prev_instr = 0
            prev_phase = -1
            for i in data.index:
                if data.at[i, "Phase"] != prev_phase:
                    cumm_instr += prev_instr

                prev_instr = data.at[i, "Instrs"]
                data.at[i, "Instrs"] += cumm_instr
                prev_phase = data.at[i, "Phase"]
                data.at[i, "Phase"] = 0

            cumm_instr = 0
            prev_instr = 0
            prev_phase = -1
            for i in data2.index:
                if data2.at[i, "Phase"] != prev_phase:
                    cumm_instr += prev_instr

                prev_instr = data2.at[i, "Instrs"]
                data2.at[i, "Instrs"] += cumm_instr
                prev_phase = data2.at[i, "Phase"]
                data2.at[i, "Phase"] = 0

            if data3 is not None:
                cumm_instr = 0
                prev_instr = 0
                prev_phase = -1
                for i in data3.index:
                    if data3.at[i, "Phase"] != prev_phase:
                        cumm_instr += prev_instr

                    prev_instr = data3.at[i, "Instrs"]
                    data3.at[i, "Instrs"] += cumm_instr
                    prev_phase = data3.at[i, "Phase"]
                    data3.at[i, "Phase"] = 0
            '''
            data = data.assign(Phase=0)
            data2 = data2.assign(Phase=0)
            if data3 is not None:
                data3 = data3.assign(Phase=0)
            args.phase[0] = 0
            csv_estimate_postfix = ""
            t_e = time.time()

            #print("Conversion took {} seconds..".format(t_e - t_s))


        prev_phase_last_ts = -1
        empty_phase_encountered = False
        # Iterate all phases, but if explicitly requested discard others
        if args.phase is not None:
            phase_list = args.phase
        else:
            phase_list = range(data["Phase"].iloc[-1] + 1)

        for phase in phase_list:
            condition = data.Phase == int(phase)
            pdata = data[condition]
            condition = data2.Phase == int(phase)
            pdata2 = data2[condition]
            pdata3 = None
            if data3 is not None:
                condition = data3.Phase == int(phase)
                pdata3 = data3[condition]
                if len(pdata3) == 0:
                    pdata3 = None
                else:
                    t_measured_all += (pdata3["Timestamp"].iloc[-1] - pdata3["Timestamp"].iloc[0])

            if len(pdata) == 0 or len(pdata2) == 0:
                empty_phase_encountered = True
                if args.verbose:
                    print("{}: WARNING: no data available in phase {}".format(args.input[0], phase))
                continue

            if prev_phase_last_ts > -1 and empty_phase_encountered and args.estimate:
                t_all += (pdata["Timestamp"].iloc[0] - prev_phase_last_ts)
                prev_phase_last_ts = -1
                empty_phase_encountered = False

            if args.verbose:
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}, mem/instr: {}".format(
                    args.input[0], len(pdata), phase, pdata["Timestamp"].iloc[0], pdata["Timestamp"].iloc[-1],
                        pdata["Timestamp"].iloc[-1] - pdata["Timestamp"].iloc[0], pdata["Instrs"].iloc[-1],
                        len(pdata) * 8 / pdata["Instrs"].iloc[-1]))
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}, mem/instr: {}".format(
                    args.input[1], len(pdata2), phase, pdata2["Timestamp"].iloc[0], pdata2["Timestamp"].iloc[-1],
                        pdata2["Timestamp"].iloc[-1] - pdata2["Timestamp"].iloc[0], pdata2["Instrs"].iloc[-1],
                        len(pdata2) * 8 / pdata2["Instrs"].iloc[-1]))
                if pdata3 is not None:
                    print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}, mem/instr: {}".format(
                        args.input[2], len(pdata3), phase, pdata3["Timestamp"].iloc[0], pdata3["Timestamp"].iloc[-1],
                            pdata3["Timestamp"].iloc[-1] - pdata3["Timestamp"].iloc[0], pdata3["Instrs"].iloc[-1],
                            len(pdata3) * 8 / pdata3["Instrs"].iloc[-1]))

            compare_window = args.compare_window_len
            if args.compare_window_unit == "ms":
                nr_wins = int((pdata["Timestamp"].iloc[-1] - pdata["Timestamp"].iloc[0]) / compare_window)
                compare_window2 = int((pdata2["Timestamp"].iloc[-1] - pdata2["Timestamp"].iloc[0]) / nr_wins)
                if pdata3 is not None:
                    compare_window3 = int((pdata3["Timestamp"].iloc[-1] - pdata3["Timestamp"].iloc[0]) / nr_wins)

            elif args.compare_window_unit == "instrs":
                nr_wins = int((pdata["Instrs"].iloc[-1] - pdata["Instrs"].iloc[0]) / compare_window)
                if nr_wins == 0:
                    nr_wins = 1
                compare_window2 = int((pdata2["Instrs"].iloc[-1] - pdata2["Instrs"].iloc[0]) / nr_wins)
                if pdata3 is not None:
                    compare_window3 = int((pdata3["Instrs"].iloc[-1] - pdata3["Instrs"].iloc[0]) / nr_wins)

            elif args.compare_window_unit == "accesses":
                nr_wins = int(len(pdata) / compare_window)
                if nr_wins == 0:
                    nr_wins = 1
                compare_window2 = int(len(pdata2) / nr_wins)
                if pdata3 is not None:
                    compare_window3 = int(len(pdata3) / nr_wins)
            else:
                print("error: invalid compare_window_unit")
                sys.exit(-1)

            if args.verbose:
                print("{} windows with compare_window: {} {}, compare_window2: {} {}".format(
                    nr_wins,
                    compare_window,
                    args.compare_window_unit,
                    compare_window2,
                    args.compare_window_unit))

            prev_win_last_ts = -1
            empty_win_encountered = False
            for win in range(nr_wins):
                pages = {}
                pages2 = {}
                inter = {}

                # Select data
                if args.compare_window_unit == "ms":
                    win_data = pdata[(pdata["Timestamp"] >= (compare_window * win)) & (pdata["Timestamp"] < (compare_window * (win + 1)))]
                    win_data2 = pdata2[(pdata2["Timestamp"] >= (compare_window2 * win)) & (pdata2["Timestamp"] < (compare_window2 * (win + 1)))]
                    if pdata3 is not None:
                        win_data3 = pdata3[(pdata3["Timestamp"] >= (compare_window3 * win)) & (pdata3["Timestamp"] < (compare_window3 * (win + 1)))]

                elif args.compare_window_unit == "instrs":
                    if win == nr_wins - 1:
                        win_data = pdata[(pdata["Instrs"] >= (compare_window * win))]
                        win_data2 = pdata2[(pdata2["Instrs"] >= (compare_window2 * win))]
                        if pdata3 is not None:
                            win_data3 = pdata3[(pdata3["Instrs"] >= (compare_window3 * win))]
                    else:
                        win_data = pdata[(pdata["Instrs"] >= (compare_window * win)) & (pdata["Instrs"] < (compare_window * (win + 1)))]
                        win_data2 = pdata2[(pdata2["Instrs"] >= (compare_window2 * win)) & (pdata2["Instrs"] < (compare_window2 * (win + 1)))]
                        if pdata3 is not None:
                            win_data3 = pdata3[(pdata3["Instrs"] >= (compare_window3 * win)) & (pdata3["Instrs"] < (compare_window3 * (win + 1)))]

                elif args.compare_window_unit == "accesses":
                    if win == nr_wins - 1:
                        win_data = pdata[(compare_window * win):]
                        win_data2 = pdata2[(compare_window2 * win):]
                        if pdata3 is not None:
                            win_data3 = pdata3[(compare_window3 * win):]
                    else:
                        win_data = pdata[(compare_window * win):(compare_window * (win + 1))]
                        win_data2 = pdata2[(compare_window2 * win):(compare_window2 * (win + 1))]
                        if pdata3 is not None:
                            win_data3 = pdata3[(compare_window3 * win):(compare_window3 * (win + 1))]

                #print("len(win_data): {}, len(win_data2): {}, len(win_data3): {}".format(len(win_data), len(win_data2), len(win_data3)))

                if len(win_data) == 0 or len(win_data2) == 0:
                    empty_win_encountered = True
                    if args.verbose:
                        print("{}: WARNING: no data available in phase {} window {}".format(args.input[0], phase, win))
                    continue

                if prev_win_last_ts > -1 and empty_win_encountered and args.estimate:
                    t_all += (win_data["Timestamp"].iloc[0] - prev_win_last_ts)
                    prev_win_last_ts = -1
                    empty_win_encountered = False

                interval_pairs = IntervalTree()

                if args.detect_intervals:
                    intervals = Memory.detect_intervals(win_data["Vaddr"].values,
                                                        args.interval_distance)
                    intervals2 = Memory.detect_intervals(win_data2["Vaddr"].values,
                                                         args.interval_distance)
                    nr_accs = []
                    for interval in intervals:
                        nr_accs.append(interval.data)

                    l_sum = 0
                    acc_limit = 0
                    for l in reversed(sorted(nr_accs)):
                        l_sum += l
                        if l_sum > len(win_data) * 0.9:
                            acc_limit = l
                            break

                    acc_limit2 = acc_limit * len(win_data) / len(win_data2)
                    #print("acc_limit: {}, acc_limit2: {}".format(acc_limit, acc_limit2))

                    for interval in sorted(intervals):
                        if interval.data < acc_limit:
                            continue

                        low_vaddr = interval.begin
                        high_vaddr = interval.end
                        interval_pages = int((high_vaddr - low_vaddr) / PAGE_SIZE)

                        for interval2 in sorted(intervals2):
                            if interval2.data < acc_limit2:
                                continue

                            low_vaddr2 = interval2.begin
                            high_vaddr2 = interval2.end
                            interval_pages2 = int((high_vaddr2 - low_vaddr2) / PAGE_SIZE)

                            if interval_pages < (interval_pages2 * 0.9) or interval_pages > (interval_pages2 * 1.1):
                                continue

                            # Don't even bother..
                            if low_vaddr == low_vaddr2:
                                break

                            if interval.data < (interval2.data * 0.8) or interval.data > (interval2.data * 1.2):
                                continue

                            interval_pairs[interval.begin:interval.end] = interval2
                            print("interval: {} - {} (#pages: {}, #accs: {}) -> {} - {} (#pages: {}, #accs: {})".format(
                                '0x%x' % interval.begin,
                                '0x%x' % interval.end,
                                interval_pages,
                                interval.data,
                                '0x%x' % interval2.begin,
                                '0x%x' % interval2.end,
                                interval_pages2,
                                interval2.data))
                            break



                    for interval in sorted(intervals):
                        if interval.data < args.min_accesses:
                            continue
                        low_vaddr = interval.begin
                        high_vaddr = interval.end
                        interval_pages = int((high_vaddr - low_vaddr) / PAGE_SIZE)
                        print("DRAM VA range: {} - {} ({} pages), # of accesses: {} ({})".format(
                            '0x%x' % interval.begin,
                            '0x%x' % interval.end,
                            interval_pages,
                            len(win_data[(win_data["Vaddr"] >= low_vaddr) & (win_data["Vaddr"] < high_vaddr)]),
                            interval.data))

                    intervals2_by_nracc = {}
                    for interval2 in sorted(intervals2):
                        if interval2.data < args.min_accesses:
                            continue
                        low_vaddr2 = interval2.begin
                        high_vaddr2 = interval2.end
                        interval_pages2 = int((high_vaddr2 - low_vaddr2) / PAGE_SIZE)
                        print("MCDRAM VA range: {} - {} ({} pages), # of accesses: {}".format(
                            '0x%x' % interval2.begin,
                            '0x%x' % interval2.end,
                            interval_pages2,
                            len(win_data2[(win_data2["Vaddr"] >= low_vaddr2) & (win_data2["Vaddr"] < high_vaddr2)])))


                # Runtime estimate?
                if args.estimate:
                    PAGE_SIZE = 4096
                    PAGE_SHIFT = 12
                    nr_hbm_accesses = 0
                    nr_accesses = 0

                    def is_hbm(nr_hbm_accesses, vaddr):
                        if hbm_perc > 0:
                            if ((vaddr >> PAGE_SHIFT) % 100 <= hbm_perc):
                                nr_hbm_accesses += 1
                        else:
                            if hbm_intervals.overlaps(vaddr):
                                nr_hbm_accesses += 1

                    t = win_data["Timestamp"].iloc[-1] - win_data["Timestamp"].iloc[0]
                    t2 = win_data2["Timestamp"].iloc[-1] - win_data2["Timestamp"].iloc[0]
                    if pdata3 is not None and len(win_data3) > 0:
                        t3 = win_data3["Timestamp"].iloc[-1] - win_data3["Timestamp"].iloc[0]
                    else:
                        t3 = 0

                    if t > (t2 * 1.03):
                        nr_accesses = len(win_data)
                        hbm_page_accs = {}
                        page_accs = {}

                        #win_t_s = time.time()
                        #print("iterating phase {}/{} window {}".format(phase, data["Phase"].iloc[-1], win))

                        # Basic iteration
                        #for i in range(len(win_data)):
                        #    if hbm_intervals.overlaps(win_data["Vaddr"].iloc[i]):
                        #        nr_hbm_accesses += 1

                        # pandas iterrows()
                        #for __ind, row in win_data.iterrows():
                        #    if hbm_intervals.overlaps(row["Vaddr"]):
                        #        nr_hbm_accesses += 1

                        # Lambda
                        #win_data.apply(lambda row: is_hbm(nr_hbm_accesses, row["Vaddr"]), axis = 1)

                        # numpy array
                        for vaddr in win_data["Vaddr"].values:
                            page_addr = vaddr & (~(PAGE_SIZE - 1))
                            if ((hbm_perc > 0 and ((page_addr >> PAGE_SHIFT) % 100 <= hbm_perc)) or
                                    hbm_intervals.overlaps(page_addr)):
                                nr_hbm_accesses += 1

                                if page_addr in hbm_page_accs:
                                    hbm_page_accs[page_addr] += 1
                                else:
                                    hbm_page_accs[page_addr] = 1
                            else:
                                if page_addr in page_accs:
                                    page_accs[page_addr] += 1
                                else:
                                    page_accs[page_addr] = 1

                        nr_accesses = 0
                        nr_hbm_accesses = 0.0
                        for p in hbm_page_accs.values():
                            nr_hbm_accesses += (float(args.hbm_factor) * p)

                        for p in page_accs.values():
                            nr_accesses += p

                        nr_accesses += nr_hbm_accesses


                        #print("iterating phase {}/{} window {} DONE, took: {} secs".format(phase, data["Phase"].iloc[-1], win, time.time() - win_t_s))

                        prev_phase_last_ts = win_data["Timestamp"].iloc[-1]
                        prev_win_last_ts = win_data["Timestamp"].iloc[-1]

                        t_orig = t
                        if t2 < t:
                            t = t - (float(t - t2) * (float(nr_hbm_accesses) / nr_accesses))
                        if args.verbose:
                            if pdata3 is not None:
                                print("phase: {}, window: {}: t: {}, t2: {}, t_measured: {}, t_estimated: {}, nr_hbm_accesses: {}, nr_accesses: {}".format(
                                    phase, win, t_orig, t2, t3, t, nr_hbm_accesses, nr_accesses))
                            else:
                                print("phase: {}, window: {}: t: {}, t2: {}, t_est: {}, nr_hbm_accesses: {}, nr_accesses: {}".format(
                                    phase, win, t_orig, t2, t, nr_hbm_accesses, nr_accesses))
                            print("")

                    t_all += t

                # Trace comparison
                else:
                    nr_hbm_accesses = 0
                    nr_accesses = 0
                    # Compare pages
                    for addr in win_data["Vaddr"].values:
                        if args.hbm:
                            if hbm_perc > 0:
                                if ((addr >> PAGE_SHIFT) % 100) <= hbm_perc:
                                    nr_hbm_accesses += 1
                            else:
                                if hbm_intervals.overlaps(addr):
                                    nr_hbm_accesses += 1

                        nr_accesses += 1

                        # Translate addr based on interval pairs if match exists
                        matches = interval_pairs[addr]
                        if len(list(matches)) == 1:
                            match = list(matches)[0]
                            taddr = addr - match.begin + match.data.begin
                            #print("address {} translated to {}".format('0x%x' % addr, '0x%x' % taddr))
                            pages[int(taddr / PAGE_SIZE)] = None
                        else:
                            pages[int(addr / PAGE_SIZE)] = None

                    for i in range(len(win_data2)):
                        pages2[int(win_data2["Vaddr"].iloc[i] / PAGE_SIZE)] = None

                    nr_overlap = 0
                    nr_total = 0
                    for page in pages.keys():
                        if page in pages2:
                            nr_overlap += 1
                            inter[page] = None
                        nr_total += 1

                    '''
                    for page in pages2.keys():
                        if page in inter:
                            continue

                        if page in pages:
                            nr_overlap += 1
                        nr_total += 1
                    '''

                    if args.compare_window_unit == "ms":
                        print("[{}, {}]({}): {} of accesses".format(win_data["Timestamp"].iloc[0], win_data["Timestamp"].iloc[-1], args.compare_window_unit, len(win_data)))
                        print("[{}, {}]({}): {} of accesses".format(win_data2["Timestamp"].iloc[0], win_data2["Timestamp"].iloc[-1], args.compare_window_unit, len(win_data2)))
                    elif args.compare_window_unit == "instrs":
                        print("[{}, {}]({}): {} of accesses".format(win_data["Instrs"].iloc[0], win_data["Instrs"].iloc[-1], args.compare_window_unit, len(win_data)))
                        print("[{}, {}]({}): {} of accesses".format(win_data2["Instrs"].iloc[0], win_data2["Instrs"].iloc[-1], args.compare_window_unit, len(win_data2)))
                    elif args.compare_window_unit == "accesses":
                        print("[{}, {}]({}): {} of accesses".format(win * compare_window, (win + 1) * compare_window - 1, args.compare_window_unit, len(win_data)))
                        print("[{}, {}]({}): {} of accesses".format(win * compare_window2, (win + 1) * compare_window2 - 1, args.compare_window_unit, len(win_data2)))

                    print("overlap: {}".format(float(nr_overlap) / nr_total))
                    if args.hbm:
                        print("HBM ranges' overlap with DRAM accesses: {}".format(nr_hbm_accesses / nr_accesses))
                    print("")

        if args.estimate:
            if not args.csv:
                if t_measured_all > 0:
                    print("Estimated overall runtime{}: {} msecs, measured: {}".format(" in phase {}".format(args.phase) if args.phase else "", t_all, t_measured_all))
                else:
                    print("Estimated overall runtime{}: {} msecs".format(" in phase {}".format(args.phase) if args.phase else "", t_all))
            else:
                est_ratio = -1
                if len(args.input) == 3:
                    inputs = os.path.basename(args.input[2]).split("-")
                    hbm_postfix = ""
                    if hbm_perc > 0:
                        hbm_postfix = "+HBM-{}".format(hbm_perc)
                    else:
                        for interval in sorted(hbm_intervals):
                            hbm_postfix += "+HBM-{}-{}".format('0x%x' % interval.begin, '0x%x' % interval.end) 
                    print("{},{},{},{},".format(inputs[1].capitalize(), "Measured", inputs[0] + hbm_postfix, "%.2f" % data3["Timestamp"].iloc[-1]))
                    est_ratio = float(data3["Timestamp"].iloc[-1]) / t_all

                inputs = os.path.basename(args.input[0]).split("-")
                hbm_postfix = ""
                if hbm_perc > 0:
                    hbm_postfix = "+HBM-{}".format(hbm_perc)
                else:
                    for interval in sorted(hbm_intervals):
                        hbm_postfix += "+HBM-{}-{}".format('0x%x' % interval.begin, '0x%x' % interval.end) 
                if est_ratio == -1:
                    print("{},{},{},{},".format(inputs[1].capitalize(), "Estimated" + csv_estimate_postfix, inputs[0] + hbm_postfix, "%.2f" % t_all))
                else:
                    print("{},{},{},{},{}".format(inputs[1].capitalize(), "Estimated" + csv_estimate_postfix, inputs[0] + hbm_postfix, "%.2f" % t_all, est_ratio))


        sys.exit(0)




    if (args.phase is not None):
        if (args.phase[0] > data["Phase"].iloc[-1]):
            print("error: phase requested..")
            sys.exit(-1)

        if len(args.phase) == 1:
            data = data[data["Phase"] == int(args.phase[0])]
        else:
            for p in range(args.phase[0], args.phase[-1] + 1):
                if p not in args.phase:
                    print("error: multipe phases must cover a contiguous range")
                    sys.exit(1)
            data = data[(data["Phase"] >= int(args.phase[0])) & (data["Phase"] <= int(args.phase[-1]))]

        if len(data) == 0:
            print("error: no data available in phase {}".format(args.phase[0]))
            sys.exit(-1)

        print("Using {} accesses in phase(s) {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
            len(data), args.phase, data["Timestamp"].iloc[0], data["Timestamp"].iloc[-1],
                data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0], data["Instrs"].iloc[-1]))

        if args.phase_window is not None or args.phase_steps is not None:
            prev_ts = data["Timestamp"].iloc[0]
            prev_instr = 0
            window = 20.0
            if args.phase_window is not None:
                window = float(args.phase_window)

            if args.phase_steps is not None:
                window = float(data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0]) / args.phase_steps

            for i in range(len(data)):
                ts = data["Timestamp"].iloc[i]
                instr = data["Instrs"].iloc[i]
                if prev_ts + window <= ts:
                    print("{}-{}: {} instrs, {} instrs / msec".format(prev_ts, ts, int(instr - prev_instr), float(instr - prev_instr) / (ts - prev_ts)))
                    prev_instr = instr
                    prev_ts = ts


    if args.phases:
        for p in range(data["Phase"].iloc[-1] + 1):
            condition = data["Phase"] == p
            __data = data[condition]
            if len(__data) == 0:
                continue

            print("Phase {}: {} accesses between {} and {} ({}) msecs, retired instructions: {}".format(
                        p,
                        len(__data),
                        int(__data["Timestamp"].iloc[0]),
                        int(__data["Timestamp"].iloc[-1]),
                        float(__data["Timestamp"].iloc[-1] - __data["Timestamp"].iloc[0]),
                        __data["Instrs"].iloc[-1]))

        sys.exit(0)




    if (args.start_timestamp is not None and
        args.end_timestamp is not None):

        if (args.start_timestamp < data["Timestamp"].iloc[0] or
            args.end_timestamp > data["Timestamp"].iloc[-1]):
            print("error: invalid time window requested..")
            sys.exit(-1)

        data = data[data["Timestamp"] > int(args.start_timestamp)]
        data = data[data["Timestamp"] < int(args.end_timestamp)]
        print("Using {} accesses in time window between {} and {} msecs".format(
            len(data), args.start_timestamp, args.end_timestamp))

    # Handle virtual range
    if (args.low_vaddr is not None and
        args.high_vaddr is not None):
        print("Low vaddr: {}, high vaddr: {}".format(
            '0x%x' % int(args.low_vaddr),
            '0x%x' % int(args.high_vaddr)))
        #data = data[data["Vaddr"] > 46913970000000][data["Vaddr"] < 46916000000000]
        data = data[data["Vaddr"] > args.low_vaddr]
        data = data[data["Vaddr"] < args.high_vaddr]

    if args.range is not None:
        print(args.range)
        sys.exit(0)

    #print(data)
    #print(data["Vaddr"].to_numpy())

    # K-Means based clustering
    if args.detect_clusters:
        print("Detecting address clusters...")
        sse = []
        kmeanss = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data["Vaddr"].to_numpy().reshape(-1, 1))
            sse.append(kmeans.inertia_)
            kmeanss.append(kmeans)
            #print("Centers for {} clusters: {}".format(k, vhex(int(kmeans.cluster_centers_))))

        kl = KneeLocator(range(1, 10), sse, curve="convex", direction="decreasing")
        nr_centers = kl.elbow

        if args.nr_clusters is not None:
            nr_centers = args.nr_clusters

        kmeans = kmeanss[nr_centers - 1]
        print("Number of clusters: {}".format(nr_centers))
        print("Cluster centers: ")
        cluster_centers_ = kmeans.cluster_centers_
        clusters = {}
        for vi in range(0, nr_centers):
            vaddr = int(cluster_centers_[vi][0])
            print("Vaddr: {}".format('0x%x' % vaddr))
            clusters[vaddr] = {
                "low" : data["Vaddr"].to_numpy().max(),
                "high" : data["Vaddr"].to_numpy().min()}

        print("Calculating cluster borders..")
        indeces = kmeans.predict(data["Vaddr"].to_numpy().reshape(-1, 1))
        print(indeces)

        vaddrlist = data["Vaddr"].tolist()
        for i in range(0, len(vaddrlist)):
            vaddr = vaddrlist[i]
            cvaddr = int(cluster_centers_[indeces[i]][0])
            if vaddr < clusters[cvaddr]["low"]:
                clusters[cvaddr]["low"] = vaddr

            if vaddr > clusters[cvaddr]["high"]:
                clusters[cvaddr]["high"] = vaddr

            vaddr = int(cluster_centers_[args.cluster_id][0])
            low_vaddr = vaddr - (args.cluster_width * 4096)
            high_vaddr = vaddr + (args.cluster_width * 4096)
            print("Using virtual range: {} - {}".format('0x%x' % low_vaddr, '0x%x' % high_vaddr))
            print(vhex(data["Vaddr"].to_numpy()))
            data = data[data["Vaddr"] > low_vaddr]
            print(vhex(data["Vaddr"].to_numpy()))
            data = data[data["Vaddr"] < high_vaddr]
            print(vhex(data["Vaddr"].to_numpy()))

    # Interval tree based VM ranges
    if args.detect_intervals:
        Memory.detect_intervals(data["Vaddr"].values,
                                args.interval_distance, verbose=True)
        sys.exit(0)


    if args.csv and len(args.input) == 1:
        inputs = os.path.basename(args.input[0]).split("-")
        hbm_postfix = ""
        for i in range(len(inputs)):
            if inputs[i] == "HBM":
                hbm_postfix += "+HBM-{}-{}".format(inputs[i+1], inputs[i+2])
        print("{},{},{},{},".format(inputs[1].capitalize(), "Measured", inputs[0] + hbm_postfix, "%.2f" % data["Timestamp"].iloc[-1]))


if __name__ == "__main__":
    main()
