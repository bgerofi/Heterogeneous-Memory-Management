#/usr/bin/python
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse
from sklearn.cluster import KMeans
from kneed import KneeLocator
from intervaltree import Interval, IntervalTree
import importlib

fig = None
axes = None
args = None

def to_hex(x, pos):
    return '0x%x' % int(x)

def auto_int(x):
    return int(x, 0)

def plot_to_file(data, output):
    print("Generating plot...")
    #plt.xticks(rotation=45, size=16)
    #plt.yticks(size=16)
    #plt.rcParams["figure.figsize"] = (8, 4)
    plt.figure(figsize=(20,5))
    ax = sns.histplot(data, x="Timestamp", y="Vaddr", bins=250);
    #ax.set_yscale('log')
    #fig, ax = plt.subplots()
    #ax.bar(data[0], data[1], color='green')
    #ax.set_xlabel('Duration')
    ax.set_xlabel('Time (msec)', size=18)
    #plt.locator_params(axis="x", nbins=1000)
    ax.set_ylabel('Virtual address', size=18)
    #plt.ylim(fom_min - 0.0025, fom_max + 0.0025)

    fmt = ticker.FuncFormatter(to_hex)
    axes = plt.gca()
    #axes.get_yaxis().set_major_locator(ticker.MultipleLocator(1))
    axes.get_yaxis().set_major_formatter(fmt)

    print("Saving to: {}".format(output))
    plt.tight_layout()
    plt.savefig(output)
    #plt.close()

    importlib.reload(matplotlib)
    importlib.reload(sns)



def plot_to_file_init(nr_rows, height_ratios, msecs):
    global fig
    global axes

    if nr_rows == 0:
        return

    print("Initializing plot for {} subplots...".format(nr_rows))
    fig, axes = plt.subplots(nr_rows, 1, figsize=(4 + int(2 * msecs / 1000), 10), sharex=True,
        gridspec_kw = {'height_ratios': height_ratios})
    if nr_rows == 1:
        axes = [axes]


def plot_to_file_add_subplot(row, data, interval_len, msecs_total):
    global fig
    global axes
    #print("Generating subplot for row {}...".format(row))

    sns.histplot(ax=axes[row], data=data, x="Timestamp", y="Vaddr", bins=150);
    #ax.set_yscale('log')
    #fig, ax = plt.subplots()
    #ax.bar(data[0], data[1], color='green')
    #ax.set_xlabel('Duration')
    #plt.locator_params(axis="x", nbins=1000)
    #axes[row].set_ylabel('vaddr ({})'.format(int(interval_len / 4096)), size=16)
    axes[row].set_ylabel('{}'.format(int(interval_len / 4096)), size=16)
    axes[row].set_xlabel('Time (msecs) [{} msecs]'.format(int(msecs_total)), size=16)
    #plt.ylim(fom_min - 0.0025, fom_max + 0.0025)

    fmt = ticker.FuncFormatter(to_hex)
    #axes.get_yaxis().set_major_locator(ticker.MultipleLocator(1))
    axes[row].get_yaxis().set_major_formatter(fmt)


def plot_to_file_finalize(output):
    print("Saving to: {}".format(output))
    plt.tight_layout()
    plt.savefig(output)
    #plt.close()

    importlib.reload(matplotlib)
    importlib.reload(sns)


def detect_intervals(data, verbose = True):
    PAGE_SIZE = 8192
    def plus(a, b):
        return a + b

    intervals = IntervalTree()

    if args.interval_distance is None:
        print("error: you must specify --interval-distance when detecting intervals")
        sys.exit(-1)

    if verbose:
        print("Detecting address intervals...")
    vaddrlist = data["Vaddr"].tolist()
    for i in range(0, len(vaddrlist)):
        vaddr = vaddrlist[i]
        # Align to page size
        low_vaddr = (vaddr & (~(PAGE_SIZE - 1))) - (args.interval_distance * PAGE_SIZE)
        high_vaddr = (vaddr + (args.interval_distance * PAGE_SIZE)) & (~(PAGE_SIZE - 1))
        intervals[low_vaddr:high_vaddr] = 1
        intervals.merge_overlaps(data_reducer = plus)

    nr_valid_intervals = 0
    height_ratios = []
    if args.plot:
        for interval in sorted(intervals):
            low_vaddr = interval.begin
            high_vaddr = interval.end
            plotdata = data.copy()
            plotdata = plotdata[plotdata["Vaddr"] >= low_vaddr]
            plotdata = plotdata[plotdata["Vaddr"] <= high_vaddr]
            if len(plotdata) < args.min_accesses:
                continue

            nr_valid_intervals += 1
            height_ratios.insert(0, high_vaddr - low_vaddr)

        plot_to_file_init(nr_valid_intervals, height_ratios, (data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0]))

    if verbose or args.plot:
        ind = 1
        for interval in sorted(intervals):
            low_vaddr = interval.begin
            high_vaddr = interval.end
            plotdata = data.copy()
            plotdata = plotdata[plotdata["Vaddr"] >= low_vaddr]
            plotdata = plotdata[plotdata["Vaddr"] <= high_vaddr]
            if len(plotdata) < args.min_accesses:
                continue

            if verbose:
                print("Virtual range: {} - {} ({} pages), nr accesses: {}".format(
                    '0x%x' % interval.begin,
                    '0x%x' % interval.end,
                    (interval.end - interval.begin) / 4096,
                    len(plotdata)))

            if args.plot:
                plot_to_file_add_subplot(nr_valid_intervals - ind,
                        plotdata, high_vaddr - low_vaddr,
                        (data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0]))
            ind += 1

    if args.plot and nr_valid_intervals > 0:
        outfile = args.input[0]

        if (args.start_timestamp is not None and
            args.end_timestamp is not None):
            outfile = "{}-{}-{}".format(outfile, args.start_timestamp, args.end_timestamp)

        if args.phase is not None:
            outfile = "{}-phase-{}".format(outfile, args.phase)
        outfile = "{}-intervals.pdf".format(outfile)
        plot_to_file_finalize(outfile)

    return intervals


def main():
    global args

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
        type=auto_int,
        help='Low virtual address.')
    parser.add_argument('-H', '--high-vaddr', required=False,
        type=auto_int,
        help='High virtual address.')

    parser.add_argument('--phases', default=False, action='store_true',
        help='Print a summary of each application phase.')
    parser.add_argument('--phase', required=False, type=int,
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

    parser.add_argument('--plot', default=False, action='store_true', help="Generate plots.")
    parser.add_argument('--min-accesses', required=False, default=100,
        type=int, help='Minimum number of accesses in a cluster/interval to plot.')

    parser.add_argument("--hbm", type=str, required=False,
        action='append',
        help=("Memory ranges that are placed in high-bandwidth memory (e.g., 0x2aaaf9144000-0x2aaafabd0000)"))
    parser.add_argument('--estimate', default=False, action='store_true',
        help=("Estimate runtime based on two traces and the specified HBM ranges."))
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--csv', default=False, action='store_true',
        help="Output results in CSV format")

    args = parser.parse_args()

    #sns.set_theme()
    sns.set_style("whitegrid")

    data = None
    data2 = None

    if not args.csv:
        print("Loading from {}...".format(args.input[0]))
    data = pd.read_feather(args.input[0])

    # Handle time window
    #data = data.set_index("Nodes")
    data["Timestamp"] = data["Timestamp"].div(args.cpu_cycles_per_ms)
    if not args.csv:
        print("Loaded {} accesses ({} phases) between {} and {} msecs from {}".format(
            len(data),
            data["Phase"].iloc[-1],
            data["Timestamp"].iloc[0],
            data["Timestamp"].iloc[-1],
            args.input[0]))

    if len(args.input) > 1:
        if not args.csv:
            print("Loading from {}...".format(args.input[1]))
        data2 = pd.read_feather(args.input[1])
        data2["Timestamp"] = data2["Timestamp"].div(args.cpu_cycles_per_ms)
        if not args.csv:
            print("Loaded {} accesses between {} and {} msecs from {}".format(
                len(data2),
                data2["Timestamp"].iloc[0],
                data2["Timestamp"].iloc[-1],
                args.input[1]))

    if len(args.input) > 2:
        if not args.csv:
            print("Loading from {}...".format(args.input[2]))
        data3 = pd.read_feather(args.input[2])
        data3["Timestamp"] = data3["Timestamp"].div(args.cpu_cycles_per_ms)
        if not args.csv:
            print("Loaded {} accesses between {} and {} msecs from {}".format(
                len(data3),
                data3["Timestamp"].iloc[0],
                data3["Timestamp"].iloc[-1],
                args.input[2]))

    if args.compare_phase:
        sys.exit(0)


    hbm_intervals = IntervalTree()
    if args.hbm:
        for i in range(len(args.hbm)):
            addrs_s = args.hbm[i].split("-")
            if len(addrs_s) < 2:
                print("error: invalid --hbm argument: {}".format(args.hbm[i]))
                sys.exit(-1)

            low = auto_int(addrs_s[0])
            high = auto_int(addrs_s[1])
            hbm_intervals[low:high] = None
        hbm_intervals.merge_overlaps()
        if not args.csv:
            for interval in sorted(hbm_intervals):
                print("HBM range: {}-{}".format(
                            '0x%x' % interval.begin,
                            '0x%x' % interval.end))

    if len(args.input) == 3:
        inputs = os.path.basename(args.input[2]).split("-")
        for i in range(len(inputs)):
            if inputs[i] == "HBM":
                low = auto_int(inputs[i+1])
                high = auto_int(inputs[i+2])
                hbm_intervals[low:high] = None

        if not args.csv:
            for interval in sorted(hbm_intervals):
                print("HBM range: {}-{}".format(
                            '0x%x' % interval.begin,
                            '0x%x' % interval.end))


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

    # Traces comparison
    if args.compare:
        # Estimated overall runtime (if estimating)
        t_all = 0
        PAGE_SIZE = 32768
        PAGE_SIZE = 65536
        if len(args.input) < 2:
            print("error: you must specify two traces when comparing..")
            sys.exit(-1)

        if data2 is None:
            print("error: you must specify two input files for comparison")
            sys.exit(-1)

        prev_phase_last_ts = -1
        empty_phase_encountered = False
        # Iterate all phases, but if explicitly requested discard others
        for phase in range(data["Phase"].iloc[-1] + 1):
            if args.phase and phase != args.phase:
                continue

            pdata = data[data["Phase"] == int(phase)]
            pdata2 = data2[data2["Phase"] == int(phase)]

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
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
                    args.input[0], len(pdata), phase, pdata["Timestamp"].iloc[0], pdata["Timestamp"].iloc[-1],
                        pdata["Timestamp"].iloc[-1] - pdata["Timestamp"].iloc[0], pdata["Instrs"].iloc[-1]))
                print("{}: {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
                    args.input[1], len(pdata2), phase, pdata2["Timestamp"].iloc[0], pdata2["Timestamp"].iloc[-1],
                        pdata2["Timestamp"].iloc[-1] - pdata2["Timestamp"].iloc[0], pdata2["Instrs"].iloc[-1]))

            compare_window = args.compare_window_len
            if args.compare_window_unit == "ms":
                nr_wins = int((pdata["Timestamp"].iloc[-1] - pdata["Timestamp"].iloc[0]) / compare_window)
                compare_window2 = int((pdata2["Timestamp"].iloc[-1] - pdata2["Timestamp"].iloc[0]) / nr_wins)

            elif args.compare_window_unit == "instrs":
                nr_wins = int((pdata["Instrs"].iloc[-1] - pdata["Instrs"].iloc[0]) / compare_window)
                compare_window2 = int((pdata2["Instrs"].iloc[-1] - pdata2["Instrs"].iloc[0]) / nr_wins)

            elif args.compare_window_unit == "accesses":
                if compare_window > (len(pdata)):
                    nr_wins = 1
                    compare_window2 = len(pdata2)
                else:
                    nr_wins = int(len(pdata) / compare_window)
                    compare_window2 = int(len(pdata2) / nr_wins)
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

            for win in range(nr_wins):
                pages = {}
                pages2 = {}
                inter = {}

                # Select data
                if args.compare_window_unit == "ms":
                    win_data = pdata[(pdata["Timestamp"] >= (compare_window * win)) & (pdata["Timestamp"] < (compare_window * (win + 1)))]
                    win_data2 = pdata2[(pdata2["Timestamp"] >= (compare_window2 * win)) & (pdata2["Timestamp"] < (compare_window2 * (win + 1)))]
                elif args.compare_window_unit == "instrs":
                    win_data = pdata[(pdata["Instrs"] >= (compare_window * win)) & (pdata["Instrs"] < (compare_window * (win + 1)))]
                    win_data2 = pdata2[(pdata2["Instrs"] >= (compare_window2 * win)) & (pdata2["Instrs"] < (compare_window2 * (win + 1)))]
                elif args.compare_window_unit == "accesses":
                    win_data = pdata[(compare_window * win):(compare_window * (win + 1))]
                    win_data2 = pdata2[(compare_window2 * win):(compare_window2 * (win + 1))]

                interval_pairs = IntervalTree()

                if args.detect_intervals:
                    intervals = detect_intervals(win_data, verbose = False)
                    intervals2 = detect_intervals(win_data2, verbose = False)

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
                    nr_hbm_accesses = 0
                    nr_accesses = 0

                    t2 = win_data2["Timestamp"].iloc[-1] - win_data2["Timestamp"].iloc[0]
                    t = win_data["Timestamp"].iloc[-1] - win_data["Timestamp"].iloc[0]

                    if t > (t2 * 1.03):
                        for i in range(len(win_data)):
                            if hbm_intervals.overlaps(win_data["Vaddr"].iloc[i]):
                                nr_hbm_accesses += 1
                            nr_accesses += 1

                        prev_phase_last_ts = win_data["Timestamp"].iloc[-1]

                        if args.verbose:
                            print("phase: {}, window: {}: t: {}, t2: {}, nr_hbm_accesses: {}, nr_accesses: {}".format(
                                phase, win, t, t2, nr_hbm_accesses, nr_accesses))
                        if t2 < t:
                            t = t - (float(t - t2) * (float(nr_hbm_accesses) / nr_accesses))

                    t_all += t

                # Trace comparison
                else:
                    nr_hbm_accesses = 0
                    nr_accesses = 0
                    # Compare pages
                    for i in range(len(win_data)):
                        addr = win_data["Vaddr"].iloc[i]

                        if args.hbm:
                            if hbm_intervals.overlaps(win_data["Vaddr"].iloc[i]):
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
                print("Estimated overall runtime{}: {} msecs".format(" in phase {}".format(args.phase) if args.phase else "", t_all))
            else:
                inputs = os.path.basename(args.input[0]).split("-")
                hbm_postfix = ""
                for interval in sorted(hbm_intervals):
                    hbm_postfix += "+HBM-{}-{}".format('0x%x' % interval.begin, '0x%x' % interval.end) 
                print("{},{},{},{}".format(inputs[1].capitalize(), "Estimated", inputs[0] + hbm_postfix, "%.2f" % t_all))

                if len(args.input) == 3:
                    inputs = os.path.basename(args.input[2]).split("-")
                    hbm_postfix = ""
                    for interval in sorted(hbm_intervals):
                        hbm_postfix += "+HBM-{}-{}".format('0x%x' % interval.begin, '0x%x' % interval.end) 
                    print("{},{},{},{}".format(inputs[1].capitalize(), "Measured", inputs[0] + hbm_postfix, "%.2f" % data3["Timestamp"].iloc[-1]))

        sys.exit(0)


    if args.phases:
        for p in range(data["Phase"].iloc[-1]):
            __data = data[data["Phase"] == p]
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





    if (args.phase is not None):
        if (args.phase > data["Phase"].iloc[-1]):
            print("error: phase requested..")
            sys.exit(-1)

        data = data[data["Phase"] == int(args.phase)]
        if len(data) == 0:
            print("error: no data available in phase {}".format(args.phase))
            sys.exit(-1)

        print("Using {} accesses in phase {}, time window between {} and {} ({}) msecs, retired instructions: {}".format(
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

    plotdatas = {}

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

        if args.cluster_id is None:
            for vaddr, cluster in clusters.items():
                low_vaddr = clusters[vaddr]["low"]
                high_vaddr = clusters[vaddr]["high"]
                plotdata = data.copy()
                plotdata = plotdata[plotdata["Vaddr"] >= low_vaddr]
                plotdata = plotdata[plotdata["Vaddr"] <= high_vaddr]
                plotdatas.append(plotdata)
                print("Virtual range: {} - {}, nr. of addresses: {}".format(
                    '0x%x' % low_vaddr,
                    '0x%x' % high_vaddr,
                    len(plotdata)))

        else:
            if args.cluster_id > nr_centers - 1:
                print("error: invalid cluster ID: {}".format(args.cluster_id))
                sys.exit(-1)

            if args.cluster_width is None:
                print("error: you must also specify --cluster-width when selecting cluster ID")
                sys.exit(-1)

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
        detect_intervals(data)

        sys.exit(0)



    #print(plotdatas)
    if args.plot:
        outfile = args.input[0]

        if (args.low_vaddr is not None and
            args.high_vaddr is not None):
            outfile = "{}-{}-{}".format(outfile, '0x%x' % args.low_vaddr, '0x%x' % args.high_vaddr)

        if (args.start_timestamp is not None and
            args.end_timestamp is not None):
            outfile = "{}-{}-{}".format(outfile, args.start_timestamp, args.end_timestamp)

        if args.phase is not None:
            outfile = "{}-phase-{}".format(outfile, args.phase)

        outfile = "{}.pdf".format(outfile)

        plot_to_file(data, outfile)


    if args.csv and len(args.input) == 1:
        inputs = os.path.basename(args.input[0]).split("-")
        hbm_postfix = ""
        for i in range(len(inputs)):
            if inputs[i] == "HBM":
                hbm_postfix += "+HBM-{}-{}".format(inputs[i+1], inputs[i+2])
        print("{},{},{},{}".format(inputs[1].capitalize(), "Measured", inputs[0] + hbm_postfix, "%.2f" % data["Timestamp"].iloc[-1]))


if __name__ == "__main__":
    main()
