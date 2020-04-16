import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def rewards_to_go(rewards):
    r = rewards.copy()
    r.reverse()
    r = list(np.cumsum(r))
    r.reverse()
    r = torch.tensor(r, requires_grad=False, dtype=torch.float32)
    return r


class Replay_buffer:
    """A reply buffer of size [size]"""

    def __init__(self, size):
        self.size = size
        self.keys = ["t", "state", "action", "reward", "state_next", "done"]
        self.key_types = [torch.int] + 5 * [torch.float32]

        self.buf = size * [None]
        self.counter = 0
        self.size_used = 0

    def is_full(self):
        return self.size_used == self.size

    def update(self, batch):

        for item in batch:
            assert set(item.keys()) == set(self.keys)

            self.buf[self.counter] = item
            self.counter += 1
            self.counter %= self.size

            if not self.is_full():
                self.size_used += 1

    def sample(self, size):
        """Returns a sample of size [size] in the form of a dictionary with six keys for
        each of which the value is a list of e.g. states"""

        ixs = np.random.randint(low=0, high=self.size_used, size=size)
        sample = [self.buf[ix] for ix in ixs]

        sample = {
            k: torch.tensor([s[k] for s in sample], requires_grad=False, dtype=k_type)
            for k, k_type in zip(self.keys, self.key_types)
        }
        return sample


class Mlp:
    """A multi-layer perceptron"""

    def __init__(
        self,
        hidden_sizes,
        in_size,
        out_size,
        activation=torch.relu,
        output_activation=None,
        no_grad=False,
    ):

        self.activation = activation
        self.output_activation = output_activation

        left_size = None
        right_size = None

        self.layers = []
        for h in hidden_sizes:
            left_size = right_size if right_size is not None else in_size
            right_size = h
            self.layers.append(torch.nn.Linear(left_size, right_size))

        self.layers.append(torch.nn.Linear(right_size, out_size))

        self.parameters = []
        for l in self.layers:
            self.parameters += l.parameters()

        if no_grad:
            for param in self.parameters:
                param.requires_grad = False

    def forward(self, x):
        """A forward pass through the network"""

        out = x
        for l in self.layers[:-1]:
            out = l.forward(out)
            out = self.activation(out)

        out = self.layers[-1].forward(out)

        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

    def init_to(self, nn):
        for l1, l2 in zip(self.layers, nn.layers):
            for p1, p2 in zip(l1.parameters(), l2.parameters()):
                p1.data = p2.data.clone()

    def polyak_update(self, nn, rho):
        for l1, l2 in zip(self.layers, nn.layers):
            for p1, p2 in zip(l1.parameters(), l2.parameters()):
                p1.data = rho * p1.data.clone() + (1 - rho) * p2.data.clone()

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()


class Algorithm:
    """An RL algorithm

    :param env_fn: a function which returns a gym environment

    :param log_dir: a directory to store experiment logs
    """

    def __init__(self, env_fn, log_dir):
        self.env = env_fn()
        self.log_dir = log_dir
        self.run_dir = None
        self.run_start_time = None
        self.writer = None

    def init_experiment(self):
        self.run_dir = os.path.join(self.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S"))
        tensorboard_command = f"tensorboard --logdir {self.run_dir}"
        print(f"Tensorboard_cmd: {tensorboard_command}")
        self.run_start_time = time.time()
        self.writer = SummaryWriter(log_dir=self.run_dir)
