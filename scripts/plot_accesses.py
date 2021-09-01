#/usr/bin/python
import sys
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



def plot_to_file_init(nr_rows, height_ratios):
    global fig
    global axes

    print("Initializing plot for {} subplots...".format(nr_rows))
    fig, axes = plt.subplots(nr_rows, 1, figsize=(10, 10), sharex=True,
        gridspec_kw = {'height_ratios': height_ratios})
    if nr_rows == 1:
        axes = [axes]
    

def plot_to_file_add_subplot(row, data, interval_len):
    global fig
    global axes
    print("Generating subplot for row {}...".format(row))

    sns.histplot(ax=axes[row], data=data, x="Timestamp", y="Vaddr", bins=150);
    #ax.set_yscale('log')
    #fig, ax = plt.subplots()
    #ax.bar(data[0], data[1], color='green')
    #ax.set_xlabel('Duration')
    axes[row].set_xlabel('time (msec)', size=16)
    #plt.locator_params(axis="x", nbins=1000)
    #axes[row].set_ylabel('vaddr ({})'.format(int(interval_len / 4096)), size=16)
    axes[row].set_ylabel('{}'.format(int(interval_len / 4096)), size=16)
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
        type=auto_int,
        help='Low virtual address.')
    parser.add_argument('-H', '--high-vaddr', required=False,
        type=auto_int,
        help='High virtual address.')

    parser.add_argument('--range', action='append', required=False,
        help='Virtual range(s) to process (format: start_addr-end_addr).')

    parser.add_argument('--detect-intervals', default=False, action='store_true')
    parser.add_argument('--interval-distance', required=False,
        type=int, help='Max number of pages between accesses that belong to the same interval.')

    parser.add_argument('--compare', default=False, action='store_true', help="Compare two traces.")
    parser.add_argument('--compare-window', required=False, default=50,
        type=int, help='Comparison window length.')
    parser.add_argument('--compare-window-unit', required=False, default="ms",
        help='Comparison window length unit (accesses or ms).')

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
    args = parser.parse_args()

    #sns.set_theme()
    sns.set_style("whitegrid")

    print("Loading from {}...".format(args.input[0]))
    data = pd.read_feather(args.input[0])

    # Handle time window
    #data = data.set_index("Nodes")
    data["Timestamp"] = data["Timestamp"].div(args.cpu_cycles_per_ms)
    print("Loaded {} accesses between {} and {} msecs from {}".format(
        len(data),
        data["Timestamp"].iloc[0],
        data["Timestamp"].iloc[-1],
        args.input[0]))

    if len(args.input) > 1:
        print("Loading from {}...".format(args.input[1]))
        data2 = pd.read_feather(args.input[1])
        data2["Timestamp"] = data2["Timestamp"].div(args.cpu_cycles_per_ms)
        print("Loaded {} accesses between {} and {} msecs from {}".format(
            len(data2),
            data2["Timestamp"].iloc[0],
            data2["Timestamp"].iloc[-1],
            args.input[1]))

    # Traces comparison
    if args.compare:
        PAGE_SIZE = 4096
        if len(args.input) < 2:
            print("error: you must specify two traces when comparing..")
            sys.exit(-1)

        i = 0
        i2 = 0
        i_prev = 0
        t_start = 0
        t_start2 = 0

        while i < len(data) and i2 < len(data2):
            pages = {}
            pages2 = {}
            if args.compare_window_unit == "ms":
                while (data["Timestamp"].iloc[i] - t_start) < args.compare_window:
                    pages[int(data["Vaddr"].iloc[i] / PAGE_SIZE)] = None
                    i = i + 1

                while (data2["Timestamp"].iloc[i2] - t_start2) < args.compare_window:
                    pages2[int(data2["Vaddr"].iloc[i2] / PAGE_SIZE)] = None
                    i2 = i2 + 1
            else:
                while (i - i_prev) < args.compare_window:
                    pages[int(data["Vaddr"].iloc[i] / PAGE_SIZE)] = None
                    i = i + 1

                while (i2 - i_prev) < args.compare_window:
                    pages2[int(data2["Vaddr"].iloc[i2] / PAGE_SIZE)] = None
                    i2 = i2 + 1


            nr_overlap = 0
            nr_total = 0
            for page in pages.keys():
                if page in pages2:
                    nr_overlap += 1
                nr_total += 1


            print("[{}, {}]: ".format(t_start, data["Timestamp"].iloc[i]))
            #print(pages)

            print("[{}, {}]: ".format(t_start2, data2["Timestamp"].iloc[i2]))
            #print(pages2)

            print("overlap: {}".format(float(nr_overlap) / nr_total))

            t_start = data["Timestamp"].iloc[i]
            t_start2 = data2["Timestamp"].iloc[i2]
            i_prev = i

            if data["Timestamp"].iloc[i] > 5000:
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

    intervals = IntervalTree()
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
        if args.interval_distance is None:
            print("error: you must specify --interval-distance when detecting intervals")
            sys.exit(-1)

        print("Detecting address intervals...")
        vaddrlist = data["Vaddr"].tolist()
        for i in range(0, len(vaddrlist)):
            vaddr = vaddrlist[i]
            low_vaddr = vaddr - (args.interval_distance * 4096)
            high_vaddr = vaddr + (args.interval_distance * 4096)
            intervals[low_vaddr:high_vaddr] = None
            intervals.merge_overlaps()

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

            plot_to_file_init(nr_valid_intervals, height_ratios)

        ind = 1
        for interval in sorted(intervals):
            low_vaddr = interval.begin
            high_vaddr = interval.end
            plotdata = data.copy()
            plotdata = plotdata[plotdata["Vaddr"] >= low_vaddr]
            plotdata = plotdata[plotdata["Vaddr"] <= high_vaddr]
            if len(plotdata) < args.min_accesses:
                continue

            print("Virtual range: {} - {} ({} pages), nr accesses: {}".format(
                '0x%x' % interval.begin,
                '0x%x' % interval.end,
                (interval.end - interval.begin) / 4096,
                len(plotdata)))

            if args.plot:
                plot_to_file_add_subplot(nr_valid_intervals - ind, plotdata, high_vaddr - low_vaddr)
            ind += 1

        if args.plot:
            outfile = args.input
            outfile = "{}-{}-{}".format(outfile, args.start_timestamp, args.end_timestamp)
            outfile = "{}-intervals.pdf".format(outfile)
            plot_to_file_finalize(outfile)

        sys.exit(0)



    #print(plotdatas)
    if args.plot:
        outfile = args.input

        if (args.low_vaddr is not None and
            args.high_vaddr is not None):
            outfile = "{}-{}-{}".format(outfile, '0x%x' % args.low_vaddr, '0x%x' % args.high_vaddr)

        outfile = "{}-{}-{}".format(outfile, args.start_timestamp, args.end_timestamp)
        outfile = "{}.pdf".format(outfile)

        plot_to_file(data, outfile)


if __name__ == "__main__":
    main()
