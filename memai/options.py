def add_traces_input_args(parser):
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
        "--cpu-cycles-per-ms",
        metavar="<int>",
        default=1400000,
        type=int,
        help="CPU cycles per millisecond (default: 1,400,000 for KNL).",
    )


def add_window_args(parser):
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
        "--page-size",
        metavar="<int>",
        default=13,
        type=int,
        help="The size of a page in number of bits.",
    )


def add_observation_args(parser):
    parser.add_argument(
        "--observation-rows",
        metavar="<int>",
        type=int,
        default=128,
        help=("The number of row in an observation (window of timestamp X address)."),
    )
    parser.add_argument(
        "--observation-columns",
        metavar="<int>",
        type=int,
        default=128,
        help=(
            "The number of columns in an observation (window of timestamp X address)."
        ),
    )
