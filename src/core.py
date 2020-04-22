import os
import time

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def lin_solve(apply_A_fn, b, n_iters=10):
    """Returns the vector A^{-1} b using conjugate gradients.

    A and b are lists of torch.tensors, not necessarily of the same length
    (which is why we cannot simply stack them) but they are to be thought of as
    vectors in this context.

    The implementation is based on the wiki page for this algorithm

    :param apply_A_fn: A function applying the matrix operator x -> Ax
    """

    x = [torch.zeros_like(t) for t in b]
    print(f"Running linsolve with {n_iters} steps")

    r = [(t1 - t2) for t1, t2 in zip(b, apply_A_fn(x))]
    r_norm = sum((t * t).sum() for t in r)
    p = [t.clone() for t in r]
    Ap = apply_A_fn(p)
    for _ in range(n_iters):
        # XXX Spinningup adds an epsilon to the denominator here
        alpha = r_norm / sum((t1 * t2).sum() for t1, t2 in zip(p, Ap))
        x = [t1 + alpha * t2 for t1, t2 in zip(x, p)]
        r_next = [t1 - alpha * t2 for t1, t2 in zip(r, Ap)]
        r_next_norm = sum((t * t).sum() for t in r_next)
        beta = r_next_norm / r_norm

        r = r_next
        r_norm = r_next_norm
        p = [t1 + beta * t2 for t1, t2 in zip(r, p)]
        Ap = apply_A_fn(p)

    return x


def rewards_to_go(rewards):
    r = rewards.copy()
    r.reverse()
    r = list(np.cumsum(r))
    r.reverse()
    r = torch.tensor(r, requires_grad=False, dtype=torch.float32)
    return r


def advantage_est(value_fn, episode, lambd):
    """Uses GAE to estimate the advantages for this episode"""
    states = np.array([step["state"] for step in episode])
    rewards = np.array([step["reward"] for step in episode])
    vs = (
        value_fn(torch.tensor(states, requires_grad=False, dtype=torch.float32))
        .detach()
        .numpy()
    )

    if len(vs.shape) > 0:
        v_nexts = np.concatenate([vs[1:], np.array([0])])
    else:
        vs = np.array([vs])
        v_nexts = np.array([0])

    deltas = rewards + v_nexts - vs
    advantages = len(deltas) * [None]
    const = 0
    advantage = 0
    for i, d in enumerate(deltas[-1::-1]):
        const += lambd ** i
        advantage = d + lambd * advantage
        advantages[-(i + 1)] = advantage / const

    return torch.tensor(advantages, requires_grad=False, dtype=torch.float32)


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
        in_size,
        out_size,
        hidden_sizes,
        activation=torch.relu,
        output_activation=None,
        no_grad=False,
    ):

        self.sizes = (in_size,) + hidden_sizes + (out_size,)
        self.activation = activation
        self.output_activation = output_activation
        self.no_grad = no_grad

        left_size = None
        right_size = None

        self.layers = []
        for h in hidden_sizes:
            left_size = right_size if right_size is not None else in_size
            right_size = h
            self.layers.append(torch.nn.Linear(left_size, right_size))

        self.layers.append(torch.nn.Linear(right_size, out_size))

    def parameters(self):
        for l in self.layers:
            yield from l.parameters()

    def disable_grad(self):
        for param in self.parameters():
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

    def set_params_to(self, nn):
        """Set params equal to that of another network"""
        for l1, l2 in zip(self.layers, nn.layers):
            for p1, p2 in zip(l1.parameters(), l2.parameters()):
                p1.data = p2.data.clone()

    def add_to_parameters(self, x):
        for p, x_p in zip(self.parameters(), x):
            p.data = p.data + x_p

    def polyak_update(self, nn, rho):
        for p1, p2 in zip(self.parameters(), nn.parameters()):
            p1.data = rho * p1.data.clone() + (1 - rho) * p2.data.clone()

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()


class Discrete_policy_network(Mlp):
    """Used in both vpg and trpo for example"""

    def __init__(self, env, *args, **kwargs):
        assert type(env.action_space) == gym.spaces.Discrete
        assert type(env.observation_space) == gym.spaces.box.Box
        assert len(env.observation_space.shape) == 1

        super().__init__(
            in_size=env.observation_space.shape[0],
            out_size=env.action_space.n,
            output_activation=lambda x: torch.softmax(x, dim=1),
            *args,
            **kwargs,
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-2)

    def action(self, state):
        """Returns an action for the current state"""

        state = torch.tensor(state, requires_grad=False, dtype=torch.float32)
        with torch.no_grad():
            pi = self.forward(state.unsqueeze(dim=0))
            action = torch.multinomial(pi.squeeze(), num_samples=1).squeeze()

        return action.numpy()

    def compute_policy_gradient(
        self, batch, return_type, weights_type, v_fn=None, gae_lambda=None
    ):
        """Calculate the policy gradient for this network

        :param v_fn: A value function, if given we use advantage estimation,
         otherwise we use rewards_to_go for the weights

        :param return_type: If "loss", then return the loss of the policy
         gradient, if "gradient" then return a clone of the gradient
        """
        self.zero_grad()

        assert return_type in ["loss", "gradient"]
        assert weights_type in ["rewards_to_go", "advantage_estimates"]

        loss_terms = []
        for episode in batch:

            # TODO Should I turn this for loop into a vector operation?
            log_probs = []
            for step in episode:
                probs = self.forward(
                    torch.tensor(
                        [step["state"]], requires_grad=False, dtype=torch.float32
                    )
                ).squeeze()
                # XXX This where we assume that the act_dim = 1, since we
                # use it to index [probs]
                assert len(probs.shape) == 1
                prob = probs[torch.tensor(step["action"])]
                log_probs.append(torch.log(prob))

            log_probs = torch.stack(log_probs)
            assert len(log_probs.shape) == 1

            if weights_type == "rewards_to_go":
                weights = rewards_to_go([step["reward"] for step in episode])
            elif weights_type == "advantage_estimates":
                assert v_fn is not None
                assert gae_lambda is not None
                weights = advantage_est(v_fn, episode, lambd=gae_lambda)
            else:
                assert False, "Unreachable"

            loss_term = -torch.sum(log_probs * weights)
            loss_terms.append(loss_term)

        loss_terms = torch.stack(loss_terms)
        loss = torch.mean(loss_terms)

        if return_type == "loss":
            return loss

        assert return_type == "gradient"
        # XXX Note the minus sign
        (-loss).backward()
        return [param.grad.detach().clone() for param in self.parameters()]

    def surrogate_advantage_approxim(self, pi_other, g):
        SA = sum(
            [
                (g_p * (theta1 - theta2)).sum()
                for g_p, theta1, theta2 in zip(
                    g, pi_other.parameters(), self.parameters()
                )
            ]
        )
        return SA

    def KL(self, batch, pi_other=None):

        self.zero_grad()

        # Calculate the KL divergence
        KL_terms = []
        for episode in batch:
            for step in episode:
                p = self.forward(
                    torch.tensor(
                        [step["state"]], requires_grad=False, dtype=torch.float32
                    )
                ).squeeze()

                if pi_other is None:
                    q = p.detach()
                else:
                    q = (
                        pi_other.forward(
                            torch.tensor(
                                [step["state"]],
                                requires_grad=False,
                                dtype=torch.float32,
                            )
                        )
                        .squeeze()
                        .detach()
                    )

                # If [pi_other] is None then this is zero (to rounding error)
                assert len(p.shape) == 1
                KL_terms.append(sum(p * torch.log(p / q)))

        KL = torch.stack(KL_terms).mean()
        return KL

    def KL_hessian_matmul_fn(self, batch):
        def KL_hessian_matmul(x):
            assert all(t.shape == p.shape for t, p in zip(x, self.parameters()))

            self.zero_grad()
            KL = self.KL(batch)
            KL.backward(create_graph=True)

            accum = torch.tensor(0.0, requires_grad=True)
            for t, param in zip(x, self.parameters()):
                accum = accum + (t * param.grad).sum()

            # XXX I want to clear out the gradient without adding operations to
            # the graph which change the derivative. Is below the best way to
            # do this?
            for param in self.parameters():
                param.grad = param.grad - param.grad.detach()

            accum.backward()
            return [p.grad.detach().clone() for p in self.parameters()]

        return KL_hessian_matmul

    def policy_gradient_update(self, batch, writer, epoch):
        loss = self.compute_policy_gradient(
            batch, return_type="loss", weights_type="rewards_to_go"
        )
        loss.backward()
        self.optimiser.step()
        writer.add_scalar("policy-gradient-loss", loss, epoch)


class Discrete_value_network(Mlp):
    """Used in trpo for example. The reward decay, gamma, is taken to be one"""

    def __init__(self, env, *args, **kwargs):
        assert type(env.action_space) == gym.spaces.Discrete
        assert type(env.observation_space) == gym.spaces.box.Box
        assert len(env.observation_space.shape) == 1

        super().__init__(
            in_size=env.observation_space.shape[0],
            out_size=1,
            output_activation=torch.squeeze,
            *args,
            **kwargs,
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-2)

    def simple_update(self, batch, writer, n_iter):
        self.zero_grad()

        v_loss_terms = []
        for episode in batch:
            targets = rewards_to_go([step["reward"] for step in episode])
            states = torch.tensor(
                [step["state"] for step in episode],
                requires_grad=False,
                dtype=torch.float32,
            )
            vs = self.forward(states)
            v_loss = torch.mean((vs - targets) ** 2)
            v_loss_terms.append(v_loss)

        v_loss = torch.stack(v_loss_terms).mean()
        v_loss.backward()
        self.optimiser.step()

        writer.add_scalar("value-network-loss", v_loss, n_iter)


class Continuous_policy_network(Mlp):
    """Used in ddpg for example"""

    def __init__(self, env, gamma, *args, **kwargs):
        assert type(env.action_space) == gym.spaces.box.Box
        assert len(env.action_space.shape) == 1
        assert type(env.observation_space) == gym.spaces.box.Box
        assert len(env.observation_space.shape) == 1

        super().__init__(
            in_size=env.observation_space.shape[0],
            out_size=env.action_space.shape[0],
            output_activation=torch.tanh,
            *args,
            **kwargs,
        )

        self.gamma = gamma
        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-2)

    def action(self, state, noise):
        """Returns an action for the current state"""

        state = torch.tensor(state, requires_grad=False, dtype=torch.float32)
        with torch.no_grad():
            pi = self.forward(state.unsqueeze(dim=0)).squeeze()
            action = pi + noise * torch.randn_like(pi)
            action = torch.clamp(action, -1, 1).numpy()

        return action


class Continuous_q_network(Mlp):
    """Used in ddpg for example"""

    def __init__(self, env, gamma, *args, **kwargs):
        assert type(env.action_space) == gym.spaces.box.Box
        assert len(env.action_space.shape) == 1
        assert type(env.observation_space) == gym.spaces.box.Box
        assert len(env.observation_space.shape) == 1

        super().__init__(
            in_size=env.observation_space.shape[0] + env.action_space.shape[0],
            out_size=1,
            output_activation=torch.squeeze,
            *args,
            **kwargs,
        )

        self.gamma = gamma
        self.recent_loss = None
        self.recent_mean_q_values = None

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-2)

    def ddpg_update(self, batch, target_pi_network, target_q_network):
        """Update the network with a one step look-ahead temporal difference target"""
        self.zero_grad()
        rewards, states_next, dones = (
            batch["reward"],
            batch["state_next"],
            batch["done"],
        )
        actions_next = target_pi_network.forward(states_next)
        state_actions = torch.cat([batch["state"], batch["action"]], dim=1)
        state_actions_next = torch.cat([states_next, actions_next], dim=1)
        q_next = target_q_network.forward(state_actions_next)
        q_targets = rewards + (1 - dones) * self.gamma * q_next

        q = self.forward(state_actions)
        q_loss = (q - q_targets) ** 2
        q_loss = torch.mean(q_loss)

        q_loss.backward()
        self.optimiser.step()

        self.recent_mean_q_values = q.mean().detach()
        self.recent_loss = q_loss.detach()


class Algorithm:
    """An RL algorithm

    :param env_fn: a function which returns a gym environment

    :param log_dir: a directory to store experiment logs
    """

    def __init__(self, exploration_period, env_fn, log_dir):
        self.env = env_fn()
        self.log_dir = log_dir
        self.exploration_period = exploration_period
        self.run_dir = None
        self.run_start_time = None
        self.writer = None
        self.t_experiment = None
        self.t_epoch = None
        self.t_episode = None
        self.epoch_start = None
        self.epoch_episodes = None
        self.epoch_rewards = None
        self.episode_done = None
        self.state = None
        self.episode = None

        print("Action_space:", self.env.action_space)
        print("Observation_space:", self.env.observation_space)

    def zero_grad(self):
        for nn in self.nns:
            nn.zero_grad()

    def init_experiment(self):
        self.run_dir = os.path.join(self.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S"))
        tensorboard_command = f"tensorboard --logdir {self.run_dir}"
        print(f"Tensorboard_cmd: {tensorboard_command}")
        self.run_start_time = time.time()
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.t_experiment = 0

    def init_epoch(self, epoch):
        self.epoch_start = time.time()
        self.t_epoch = 0
        self.epoch_episodes = 0
        self.epoch_rewards = []
        print(f"Epoch: {epoch}")

    def init_episode(self):
        self.t_episode = 0
        self.epoch_episodes += 1
        self.episode_done = False
        self.state = self.env.reset()
        self.episode = []

    def episode_step(self):
        self.t_episode += 1
        self.t_epoch += 1
        self.t_experiment += 1

        if (
            self.exploration_period is not None
            and self.t_experiment < self.exploration_period
        ):
            action = self.env.action_space.sample()
        else:
            action = self.action()

        state_next, reward, self.episode_done, _info = self.env.step(action)

        step = {
            "t": self.t_experiment,
            "state": self.state,
            "action": action,
            "reward": reward,
            "state_next": state_next,
            "done": self.episode_done,
        }

        self.episode.append(step)
        self.state = state_next

    def terminate_episode_early(self, max_episode_length, epoch_size):
        assert self.t_epoch <= epoch_size
        if self.t_episode > max_episode_length:
            print(
                f"Episode steps {self.t_episode} longer than max {max_episode_length}. "
                "Terminating episode early."
            )
            return True
        if self.t_epoch == epoch_size:
            print(f"Epoch size {epoch_size} reached. Terminating episode early.")
            return True
        return False

    def end_episode(self):
        self.epoch_rewards.append(sum(step["reward"] for step in self.episode))

    def end_epoch(self, epoch):
        self.writer.add_scalar("epoch-seconds", time.time() - self.epoch_start, epoch)
        self.writer.add_scalar("mean-reward", np.mean(self.epoch_rewards), epoch)
        self.writer.add_scalar("std-reward", np.std(self.epoch_rewards), epoch)
        self.writer.add_scalar("max-reward", max(self.epoch_rewards), epoch)
        self.writer.add_scalar("min-reward", min(self.epoch_rewards), epoch)
        self.writer.add_scalar("n-episodes", self.epoch_episodes, epoch)
        self.writer.add_scalar("n-epoch-interacts", self.t_epoch, epoch)
