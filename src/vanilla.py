import time

import gym
import numpy as np
import torch

import core

# TODO Turn for-loops into vectors, e.g. the log_probs


class Vanilla(core.Algorithm):
    """A vanilla policy gradient algorithm

    Finite horizon un-discounted reward. Initialised with a policy network
    """

    def __init__(self, hidden_sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert type(self.env.action_space) == gym.spaces.Discrete
        assert type(self.env.observation_space) == gym.spaces.box.Box
        assert len(self.env.observation_space.shape) == 1

        print("Action_space:", self.env.action_space)
        print("Observation_space:", self.env.observation_space)

        self.nn = core.Mlp(
            hidden_sizes=hidden_sizes,
            in_size=self.env.observation_space.shape[0],
            out_size=self.env.action_space.n,
        )

        self.optimiser = torch.optim.Adam(self.nn.parameters, lr=1e-2)
        self.start_time = None

    def zero_grad(self):
        self.nn.zero_grad()

    def pi(self, states):
        """Policy, i.e. forward pass then softmax

        :param states: list of states
        :type states: torch.tensor

        :returns: list of probability distributions (over actions) in each state
        :rtype: torch.tensor: n_states, n_actions
        """
        x = self.nn.forward(states)
        x = torch.softmax(x, dim=1)

        return x

    def action(self, state):
        """Draws an action for the given state

        :param state: The state
        :type state: torch.tensor: state_dim

        :returns: Action
        :rtype: torch.tensor, 0-dim
        """
        with torch.no_grad():
            pi = self.pi(state.unsqueeze(dim=0))
            action = torch.multinomial(pi.squeeze(), num_samples=1).squeeze()

        return action

    def train(self, epoch_size, epochs, max_episode_length):
        """Trains the algorithm"""

        self.init_experiment()
        t_experiment = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_rewards = []

            print(f"Epoch: {epoch}")

            epoch_batch = []
            n_episodes = 0
            t_epoch = 0

            while t_epoch < epoch_size:

                # Run an episode
                done = False
                state = self.env.reset()
                episode = []
                t_episode = 0

                while not done:
                    assert t_epoch <= epoch_size
                    # End the episode if it's too long
                    if t_episode > max_episode_length:
                        print(
                            f"Episode steps {t_episode} longer than max {max_episode_length}. "
                            "Terminating episode early."
                        )
                        break
                    if t_epoch == epoch_size:
                        print(
                            f"Epoch size {epoch_size} reached. Terminating episode early."
                        )
                        break

                    t_episode += 1
                    t_epoch += 1
                    t_experiment += 1

                    action = self.action(
                        torch.tensor(state, requires_grad=False, dtype=torch.float32)
                    ).numpy()

                    state_next, reward, done, _info = self.env.step(action)
                    breakpoint()

                    step = {
                        "t": t_experiment,
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "state_next": state_next,
                        "done": done,
                    }

                    episode.append(step)
                    state = state_next

                n_episodes += 1
                epoch_batch.append(episode)

                # Next episode

            # End of episodes

            # Perform update for this epoch
            self.zero_grad()

            loss_list = []
            for episode in epoch_batch:
                log_probs = []
                for step in episode:
                    probs = self.pi(
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

                weights = core.rewards_to_go([step["reward"] for step in episode])
                epoch_rewards.append(weights[0])
                loss = -torch.sum(log_probs * weights)
                loss_list.append(loss)

            loss_list = torch.stack(loss_list)
            loss = torch.mean(loss_list)
            assert loss.requires_grad
            loss.backward()
            self.optimiser.step()

            self.writer.add_scalar("epoch-seconds", time.time() - epoch_start, epoch)
            self.writer.add_scalar("mean-reward", np.mean(epoch_rewards), epoch)
            self.writer.add_scalar("std-reward", np.std(epoch_rewards), epoch)
            self.writer.add_scalar("max-reward", max(epoch_rewards), epoch)
            self.writer.add_scalar("min-reward", min(epoch_rewards), epoch)
            self.writer.add_scalar("epoch-loss", loss, epoch)
            self.writer.add_scalar("n-episodes", n_episodes, epoch)
            self.writer.add_scalar("n-epoch-interacts", t_epoch, epoch)

            # Next epoch

        # End of epochs
