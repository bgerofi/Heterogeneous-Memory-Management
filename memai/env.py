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
        page_size=20,
        interval_distance=28,
        observation_space=WindowObservationSpace(128, 128),
        action_space=MovePagesActionSpace(
            num_actions=128, move_page_cost=10, page_size=1 << 14
        ),
    ):
        # Gym specific attributes
        self.observation_space = observation_space
        self.action_space = action_space
        self._preprocessing = Preprocessing.from_pandas(
            preprocessed_file,
            page_size,
            interval_distance,
            observation_space,
        )
        self._reset()

    def _reset(self):
        self._hbm_intervals = IntervalTree()
        self.estimated_time = 0.0
        self.move_pages_time = 0.0
        self.hbm_time = 0.0
        self.ddr_time = 0.0
        self._iterator = iter(self._preprocessing)
        self._previous_input = None

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
            observation, pages, count, t_ddr, t_hbm = next(self._iterator)
            pages_access = list(zip(pages, count))
            estimated_time = Estimator.estimate_window(
                pages_access, t_ddr, t_hbm, self._hbm_intervals
            )

            self.hbm_time += t_hbm
            self.ddr_time += t_ddr
            self.move_pages_time += move_pages_time
            self.estimated_time += estimated_time
            self._previous_input = pages_access

            # The reward is negative for moving pages (we want to avoid that)
            reward = -1.0 * move_pages_time
            stop = False

        except StopIteration:
            # The reward is negative for the total elapsed time
            # (we want to minimize elapsed time).
            reward = -1.0 * self.estimated_time
            stop = True
            observation = None

        debug_info = {}
        return observation, reward, stop, debug_info

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
    add_window_args(parser)
    add_observation_args(parser)
    add_interval_args(parser)

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

    observation_space = WindowObservationSpace(
        args.observation_rows, args.observation_columns
    )

    action_space = MovePagesActionSpace(
        args.num_actions, args.move_page_cost, args.page_size
    )
    env = TraceEnv(
        args.input,
        args.page_size,
        args.interval_distance,
        observation_space=observation_space,
        action_space=action_space,
    )

    stop = False
    t = time.time()
    while not stop:
        observation, reward, stop, debug_info = env.step(action_space.sample())
    simulation_time = time.time() - t
    simulated_time = env.estimated_time + env.move_pages_time

    print("Time DDR: {:.2f} (s)".format(env.ddr_time / 1000.0))
    print("Time HBM: {:.2f} (s)".format(env.hbm_time / 1000.0))
    print("Simulated Time: {:.2f} (s)".format(simulated_time / 1000.0))
    print("\tEstimated Time: {:.2f} (s)".format(env.estimated_time / 1000.0))
    print("\tMove Pages Time: {:.2f} (s)".format(env.move_pages_time / 1000.0))
    print("Simulation Time: {:.2f} (s)".format(simulation_time))
