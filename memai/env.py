from memai import (
    Estimator,
    Memory,
    WindowObservationSpace,
    NeighborActionSpace,
    Preprocessing,
)

from gym import Env, spaces
import numpy as np
from intervaltree import IntervalTree
import tqdm


class GymEnv(Env):
    def __init__(
        self,
        preprocessed_file,
        num_actions=128,
        move_page_cost=0.01,
        hbm_size=1 << 34,
    ):

        preprocess_args = Preprocessing.parse_filename(preprocessed_file)
        # Gym specific attributes
        self.observation_space = preprocess_args["observation_space"]
        self.action_space = NeighborActionSpace(
            num_actions,
            move_page_cost,
            preprocess_args["page_size"],
        )
        self._compare_unit = preprocess_args["compare_unit"]
        self._preprocessing = Preprocessing.from_pandas(
            preprocessed_file,
            preprocess_args["page_size"],
            preprocess_args["interval_distance"],
            preprocess_args["observation_space"],
        )
        self._hbm_memory = Memory(hbm_size)
        self.progress_bar = tqdm.tqdm()
        self._reset()

    def __len__(self):
        return len(self._preprocessing)

    def _reset(self):
        self._hbm_memory.empty()
        self._estimated_times = {}
        self._window_index = 0
        self.move_pages_time = 0.0
        self._iterator = iter(self._preprocessing)
        self._previous_input = None
        self.progress_bar.reset(total=len(self._preprocessing))

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
        reward = 0
        try:
            # Do action (move pages) and retrieve move pages penalty.
            if self._previous_input is not None:
                move_pages_time = self.action_space.do_action(
                    action,
                    self._hbm_memory,
                    *self._previous_input,
                )
            else:
                move_pages_time = 0
            reward -= 1.0 * move_pages_time

            # Get next window, associated observations and estimation.
            observation, pages, count, t_ddr, t_hbm, i = next(self._iterator)
            pages_access = list(zip(pages, count))
            estimated_time = Estimator.estimate_window(
                pages_access, t_ddr, t_hbm, self._hbm_memory._chunks
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
                    reward += self._estimate_window()
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
                    self._hbm_memory._chunks.copy(),
                ]
                self._window_index = i
            self._previous_input = (observation, pages, count, t_ddr, t_hbm)

            # The reward is negative for moving pages (we want to avoid that)
            stop = False
            self.progress_bar.update(1)

        except StopIteration:
            # Don't forget to estimate last window.
            reward += self._estimate_window()
            stop = True
            observation = None
            self.progress_bar.update(len(self._preprocessing))
            self.progress_bar.clear()
        debug_info = {}
        return observation, reward, stop, debug_info

    def _estimate_window(self):
        i = self._window_index
        t_ddr, t_hbm, pages, count, intervals = self._estimated_times[i]
        estimated_time = Estimator.estimate_window(
            list(zip(pages, count)), t_ddr, t_hbm, intervals
        )
        self._estimated_times[i] = (t_ddr, t_hbm, estimated_time)
        return max([t_ddr, t_hbm]) - estimated_time

    def reset(
        self,
        *,
        seed=None,
        return_info=False,
        options=None,
    ):
        self._reset()
        try:
            # The void action when the hbm is empty.
            action = np.zeros(self.action_space.shape[0], dtype=self.action_space.dtype)
            obs, _, _, _ = self.step(action)
            return obs
        except StopIteration:
            return None

    def render(self, mode="human"):
        s = self._hbm_memory.report()
        s += "Simulated Time Summary:\n"
        s += "\tTotal Simulated Time: {:.2f} ({})\n".format(
            self.estimated_time + self.move_pages_time, self._compare_unit
        )
        s += "\tApplication Memory Access Time: {:.2f} ({})\n".format(
            self.estimated_time, self._compare_unit
        )
        s += "\tRuntime Move Pages Time: {:.2f} ({})\n".format(
            self.move_pages_time, self._compare_unit
        )
        s += "Real Execution Time Summary:\n"
        s += "\tTime DDR: {:.2f} ({})\n".format(self.t_ddr, self._compare_unit)
        s += "\tTime HBM: {:.2f} ({})\n".format(self.t_hbm, self._compare_unit)

        if mode == "ansi":
            return s
        elif mode == "human":
            print(s)
        else:
            raise ValueError("Rendering only supports 'human' or 'ansi'")


if __name__ == "__main__":
    import argparse
    import time
    from memai.options import *

    parser = argparse.ArgumentParser()
    add_env_args(parser)

    parser.add_argument(
        "--input",
        metavar="<file.feather>",
        required=True,
        type=str,
        help="A preprocessed trace obtained with the script preprocessing.py."
        "Observations shape must match the shape of observations in the trace.",
    )
    actions = ["random", "all_hbm", "all_ddr"]
    parser.add_argument(
        "--action",
        metavar="<action_kind>",
        default="random",
        type=str,
        choices=actions,
        help="The cost of moving a page in milliseconds.",
    )

    args = parser.parse_args()

    env = GymEnv(
        args.input,
        args.num_actions,
        args.move_page_cost,
        args.hbm_size << 20,
    )

    stop = False
    t = time.time()
    while not stop:
        if args.action == "random":
            action = env.action_space.sample()
        elif args.action == "all_hbm":
            action = env.action_space.all_to_hbm_action()
        elif args.action == "all_ddr":
            action = env.action_space.all_to_ddr_action()
        observation, reward, stop, debug_info = env.step(action)
    simulation_time = time.time() - t

    env.render()
    print("Simulation Time: {:.2f} (s)".format(simulation_time))
