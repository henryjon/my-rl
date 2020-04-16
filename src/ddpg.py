import time

import gym
import numpy as np
import torch

import core


class DDPG(core.Algorithm):
    """A Deep Deterministic Policy Gradient algorithm"""

    def __init__(
        self,
        hidden_sizes,
        gamma,
        rho,
        action_noise,
        replay_buffer_size,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        assert type(self.env.action_space) == gym.spaces.box.Box
        assert len(self.env.action_space.shape) == 1
        assert type(self.env.observation_space) == gym.spaces.box.Box
        assert len(self.env.observation_space.shape) == 1

        print("Action_space:", self.env.action_space)
        print("Observation_space:", self.env.observation_space)

        # XXX We assume the max action size is 1
        self.max_action_size = 1
        self.action_noise = action_noise
        self.gamma = gamma
        self.rho = rho

        # Q-network
        self.q_nn = core.Mlp(
            in_size=self.env.observation_space.shape[0]
            + self.env.action_space.shape[0],
            out_size=1,
            hidden_sizes=hidden_sizes,
        )
        # Target Q-network (no grad)
        self.q_nn_target = core.Mlp(
            in_size=self.env.observation_space.shape[0]
            + self.env.action_space.shape[0],
            out_size=1,
            hidden_sizes=hidden_sizes,
            no_grad=True,
        )

        # pi-network (output activation = tanh)
        self.pi_nn = core.Mlp(
            in_size=self.env.observation_space.shape[0],
            out_size=self.env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
            output_activation=torch.tanh,
        )

        # Target pi-network (no grad and output activation = tanh)
        self.pi_nn_target = core.Mlp(
            in_size=self.env.observation_space.shape[0],
            out_size=self.env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
            no_grad=True,
            output_activation=torch.tanh,
        )

        # Initialise the target networks to the same parameters
        self.pi_nn_target.init_to(self.pi_nn)
        self.q_nn_target.init_to(self.q_nn)

        self.replay_buffer = core.Replay_buffer(size=replay_buffer_size)
        self.q_optimiser = torch.optim.Adam(self.q_nn.parameters, lr=1e-3)
        self.pi_optimiser = torch.optim.Adam(self.pi_nn.parameters, lr=1e-3)

    def zero_grad(self):
        for nn in [self.q_nn, self.pi_nn, self.q_nn_target, self.pi_nn_target]:
            nn.zero_grad()

    def _pi_of_nn(self, nn_policy, states):
        out = nn_policy.forward(states)
        # XXX This is why we use tanh output_activation
        out = out * self.max_action_size

        return out

    def pi(self, states):
        """Returns the deterministic policy"""
        return self._pi_of_nn(self.pi_nn, states)

    def pi_target(self, states):
        """Returns the deterministic target policy"""
        return self._pi_of_nn(self.pi_nn_target, states)

    # XXX Be sure to use action_noise in the train phase
    def action(self, state, action_noise=None):
        """Draws an action for the given state

        :param state: The state
        :type state: torch.tensor: state_dim

        :param action_noise: the amount of noise to add to the actions
        :type param: float in [0, 1]

        :returns: Action
        :rtype: torch.tensor: action_dim

        """
        with torch.no_grad():
            pi = self.pi(state.unsqueeze(dim=0)).squeeze()

            if action_noise is not None:
                pi = pi + action_noise * torch.randn_like(pi)
                pi = torch.clamp(pi, -self.max_action_size, self.max_action_size)

        return pi

    def q(self, state_actions):
        """Returns the action-value network output as a vector"""
        out = self.q_nn.forward(state_actions)
        out = torch.squeeze(out)

        return out

    def q_targets(self, rewards, states_next, dones):
        actions_next = self.pi_target(states_next)
        state_actions_next = torch.cat([states_next, actions_next], dim=1)
        q_target = self.q_nn_target.forward(state_actions_next)
        q_target = torch.squeeze(q_target)
        targets = rewards + (1 - dones) * self.gamma * q_target

        return targets

    def train(
        self, epochs, epoch_size, batch_size, exploration_period, max_episode_length
    ):
        """Trains the algorithm

        :param max_episode_length: number of steps in which to initially take
         random actions
        :type max_episode_length: int
        """

        self.init_experiment()
        t_experiment = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_rewards = []
            epoch_q_values = []
            epoch_q_loss = []
            epoch_pi_loss = []

            print(f"Epoch: {epoch}")

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

                    if t_experiment < exploration_period:
                        action = self.env.action_space.sample()
                    else:
                        action = self.action(
                            torch.tensor(
                                state, requires_grad=False, dtype=torch.float32
                            ),
                            action_noise=self.action_noise,
                        ).numpy()

                    state_next, reward, done, _info = self.env.step(action)

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

                # TODO Should we be adding episodes to the buffer if they are
                # not finished?
                epoch_rewards.append(sum(step["reward"] for step in episode))
                self.replay_buffer.update(episode)

                # Do an update for each step in the episode
                # XXX Seems like a strange way to set the frequency of updates
                for _ in range(len(episode)):

                    batch = self.replay_buffer.sample(size=batch_size)
                    state_actions = torch.cat([batch["state"], batch["action"]], dim=1)
                    q_targets = self.q_targets(
                        batch["reward"], batch["state_next"], batch["done"]
                    )

                    # Update the q network
                    self.zero_grad()

                    q = self.q(state_actions)
                    q_loss = (self.q(state_actions) - q_targets) ** 2
                    q_loss = torch.mean(q_loss)

                    q_loss.backward()
                    self.q_optimiser.step()
                    epoch_q_loss.append(q_loss.detach().numpy())
                    epoch_q_values += list(q.detach().numpy())

                    # Update the policy network
                    self.zero_grad()

                    theo_actions = self.pi(batch["state"])
                    theo_state_actions = torch.cat(
                        [batch["state"], theo_actions], dim=1
                    )
                    pi_loss = -self.q(theo_state_actions)
                    pi_loss = torch.mean(pi_loss)

                    pi_loss.backward()
                    self.pi_optimiser.step()
                    epoch_pi_loss.append(pi_loss.detach().numpy())

                    # Polyak update the target networks
                    self.pi_nn_target.polyak_update(self.pi_nn, self.rho)
                    self.q_nn_target.polyak_update(self.q_nn, self.rho)

                # Next episode

            # End of episodes

            self.writer.add_scalar("epoch-seconds", time.time() - epoch_start, epoch)
            self.writer.add_scalar("mean-reward", np.mean(epoch_rewards), epoch)
            self.writer.add_scalar("std-reward", np.std(epoch_rewards), epoch)
            self.writer.add_scalar("max-reward", max(epoch_rewards), epoch)
            self.writer.add_scalar("min-reward", min(epoch_rewards), epoch)
            self.writer.add_scalar("mean-q-values", np.mean(epoch_q_values), epoch)
            self.writer.add_scalar("mean-q-loss", np.mean(epoch_q_loss), epoch)
            self.writer.add_scalar("mean-pi-loss", np.mean(epoch_pi_loss), epoch)
            self.writer.add_scalar("n-episodes", n_episodes, epoch)
            self.writer.add_scalar("n-epoch-interacts", t_epoch, epoch)

            # Next epoch

        # End of epochs
        print("Finished training")
