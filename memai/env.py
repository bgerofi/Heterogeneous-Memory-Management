from memai import (
    Estimator,
    WindowObservationSpace,
    MovePagesActionSpace,
    Preprocessing,
)

from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree


class TraceEnv(Env):
    def __init__(
        self,
        preprocessed_file,
        num_actions=128,
        move_page_cost=10,
    ):

        preprocess_args = Preprocessing.parse_filename(preprocessed_file)
        # Gym specific attributes
        self.observation_space = preprocess_args["observation_space"]
        self.action_space = MovePagesActionSpace(
            num_actions, move_page_cost, preprocess_args["page_size"]
        )
        self._compare_unit = preprocess_args["compare_unit"]
        self._preprocessing = Preprocessing.from_pandas(
            preprocessed_file,
            preprocess_args["page_size"],
            preprocess_args["interval_distance"],
            preprocess_args["observation_space"],
        )
        self._reset()

    def _reset(self):
        self._hbm_intervals = IntervalTree()
        self._estimated_times = {}
        self._window_index = 0
        self.move_pages_time = 0.0
        self._iterator = iter(self._preprocessing)
        self._previous_input = None

    @property
    def estimated_time(self):
        return sum((v[2] for v in self._estimated_times.values() if type(v) == tuple))

    @property
    def t_ddr(self):
        return sum((v[0] for v in self._estimated_times.values()))

    @property
    def t_hbm(self):
        return sum((v[1] for v in self._estimated_times.values()))

    def step(self, action):
        try:
            # Do action (move pages) and retrieve move pages penalty.
            if self._previous_input is not None:
                move_pages_time = self.action_space.do_action(
                    action, self._previous_input, self._hbm_intervals
                )
            else:
                move_pages_time = 0

            # Get next window, associated observations and estimation.
            observation, pages, count, t_ddr, t_hbm, i = next(self._iterator)
            pages_access = list(zip(pages, count))
            estimated_time = Estimator.estimate_window(
                pages_access, t_ddr, t_hbm, self._hbm_intervals
            )

            self.move_pages_time += move_pages_time
            # If we already encountered the window, we only append pages access
            # for evaluation at the end of the window.
            try:
                self._estimated_times[i][2] = np.concatenate(
                    (self._estimated_times[i][2], pages)
                )
                self._estimated_times[i][3] = np.concatenate(
                    (self._estimated_times[i][3], count)
                )
            # If this is a new window starting.
            # We estimate the time on the past window and initialize the current
            # window.
            except KeyError:
                if i > 0:  # skip first iteration.
                    self._estimate_window()
                # Initialize new window.
                # t_ddr, t_hbm and hbm mappings are not updated with other
                # observations of the same window since they all share the same.
                # hbm mappings are copied before actions on new are performed
                # such that time estimation for the window happens before the AI
                # can move pages with the knowledge of this window memory
                # accesss.
                self._estimated_times[i] = [
                    t_ddr,
                    t_hbm,
                    pages,
                    count,
                    self._hbm_intervals.copy(),
                ]
                self._window_index = i
            self._previous_input = pages_access

            # The reward is negative for moving pages (we want to avoid that)
            reward = -1.0 * move_pages_time
            stop = False

        except StopIteration:
            # Don't forget to estimate last window.
            self._estimate_window()
            # Sum all estimations into the total estimated execution time.
            estimated_time = sum((v[0] for v in self._estimated_times.values()))
            # The reward is negative for the total elapsed time
            # (we want to minimize elapsed time).
            reward = -1.0 * self.estimated_time
            stop = True
            observation = None

        debug_info = {}
        return observation, reward, stop, debug_info

    def _estimate_window(self):
        i = self._window_index
        t_ddr, t_hbm, pages, count, intervals = self._estimated_times[i]
        estimated_time = Estimator.estimate_window(
            list(zip(pages, count)), t_ddr, t_hbm, intervals
        )
        self._estimated_times[i] = (t_ddr, t_hbm, estimated_time)

    def reset(
        self,
        *,
        seed=None,
        return_info=False,
        options=None,
    ):
        super().reset(seed=seed)
        self._reset()
        try:
            # The void action when the hbm is empty.
            action = np.zeros(self.action_space.n, dtype=self.action_space.dtype)
            obs, _, _, _ = self.step(action)
            return obs
        except StopIteration:
            return None

    def render(self, mode="human"):
        total_time = self.estimated_time_ / 1000.0
        penalty_time = self.move_pages_time_ / 1000.0

        s = "Estimated time = {.2f}(s): ({.2f}(s) + {.2f}(s) penalty on page moves)".format(
            total_time + penalty_time, total_time, penalty_time
        )

        if mode == "ansi":
            return s
        elif mode == "human":
            print(s)
        else:
            raise ValueError("Rendering only supports 'human' or 'ansi'")


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc
    from memai.options import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        metavar="<file.feather>",
        required=True,
        type=str,
        help="A preprocessed trace obtained with the script preprocessing.py."
        "Observations shape must match the shape of observations in the trace.",
    )
    parser.add_argument(
        "--num-actions",
        metavar="<int>",
        default=128,
        type=int,
        help="The number of possible actions.",
    )
    parser.add_argument(
        "--move-page-cost",
        metavar="<int>",
        default=10,
        type=int,
        help="The cost of moving a page in milliseconds.",
    )

    args = parser.parse_args()

    env = TraceEnv(
        args.input,
        args.num_actions,
        args.move_page_cost,
    )

    stop = False
    t = time.time()
    while not stop:
        observation, reward, stop, debug_info = env.step(env.action_space.sample())
    simulation_time = time.time() - t
    simulated_time = env.estimated_time + env.move_pages_time

    print("Time DDR: {:.2f} ({})".format(env.t_ddr, env._compare_unit))
    print("Time HBM: {:.2f} ({})".format(env.t_hbm, env._compare_unit))
    print("Simulated Time: {:.2f} ({})".format(simulated_time, env._compare_unit))
    print("\tEstimated Time: {:.2f} ({})".format(env.estimated_time, env._compare_unit))
    print(
        "\tMove Pages Time: {:.2f} ({})".format(env.move_pages_time, env._compare_unit)
    )
    print("Simulation Time: {:.2f} (s)".format(simulation_time))
