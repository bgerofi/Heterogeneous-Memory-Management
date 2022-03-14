import pfrl
import torch
import torch.nn
import argparse
import numpy as np

from memai import Preprocessing, GymEnv
from memai.options import *


class QFunction(torch.nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_shape[0], 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)


def make_DoubleDQN_agent(environment, num_actions, gpu):
    q_func = QFunction(environment.observation_space.shape, num_actions)
    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
    # Set the discount factor that discounts future rewards.
    gamma = 0.9
    # Use epsilon-greedy for exploration
    explorer = pfrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample
    )

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**6)

    # Since observations from GymEnv is numpy.int64 while
    # As PyTorch only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=len(env) + 1,
        update_interval=1,
        target_update_interval=100,
        phi=phi,
        gpu=gpu,
    )
    return agent


def train(environment, agent):
    reset = False
    obs = environment.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, _ = environment.step(action)
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if args.model_dir is not None:
        agent.save(args.model_dir)


def evaluate(env, agent):
    with agent.eval_mode():
        reset = False
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
    simulated_time = env.estimated_time + env.move_pages_time

    print("Time DDR: {:.2f} ({})".format(env.t_ddr, env._compare_unit))
    print("Time HBM: {:.2f} ({})".format(env.t_hbm, env._compare_unit))
    print("Simulated Time: {:.2f} ({})".format(simulated_time, env._compare_unit))
    print("\tEstimated Time: {:.2f} ({})".format(env.estimated_time, env._compare_unit))
    print(
        "\tMove Pages Time: {:.2f} ({})".format(env.move_pages_time, env._compare_unit)
    )


parser = argparse.ArgumentParser()
add_env_args(parser)
parser.add_argument(
    "--gpu",
    metavar="<int>",
    default=-1,
    type=int,
    help="The GPU device id to use. Default to -1: CPU only.",
)
parser.add_argument(
    "--model-dir",
    metavar="<path>",
    default=None,
    type=str,
    help="The directory where to load and save a trained model.",
)
parser.add_argument(
    "action",
    metavar="ACTION",
    choices=["train", "eval"],
    type=str,
    help="Chose whether to train a model or to evaluate it.",
)
parser.add_argument(
    "--input",
    metavar="<file.feather>",
    required=True,
    type=str,
    action="append",
    help="A preprocessed trace obtained with the script preprocessing.py."
    "Observations shape must match the shape of observations in the trace.",
)

args = parser.parse_args()

for input_file in args.input:
    env = GymEnv(
        input_file,
        args.num_actions,
        args.move_page_cost,
        args.hbm_size << 20,
    )

    agent = make_DoubleDQN_agent(env, args.num_actions, args.gpu)

    if args.model_dir is not None:
        try:
            agent.load(args.model_dir)
        except FileNotFoundError as e:
            if args.action == "eval":
                raise e

    if args.action == "train":
        train(env, agent)
    elif args.action == "eval":
        evaluate(env, agent)

    if args.model_dir is not None:
        agent.save(args.model_dir)
