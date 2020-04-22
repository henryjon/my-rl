import numpy as np
import torch

import core


class DDPG(core.Algorithm):
    """A Deep Deterministic Policy Gradient algorithm"""

    def __init__(
        self,
        pi_hidden_sizes,
        q_hidden_sizes,
        gamma,
        rho,
        action_noise,
        replay_buffer_size,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # XXX We assume the max action size is 1
        self.max_action_size = 1
        self.action_noise = action_noise
        self.gamma = gamma
        self.rho = rho

        self.q_nn = core.Continuous_q_network(
            self.env, self.gamma, hidden_sizes=q_hidden_sizes
        )
        self.pi_nn = core.Continuous_policy_network(
            self.env, self.gamma, hidden_sizes=pi_hidden_sizes
        )
        self.q_nn_target = core.Continuous_q_network(
            self.env, self.gamma, hidden_sizes=q_hidden_sizes
        )
        self.pi_nn_target = core.Continuous_policy_network(
            self.env, self.gamma, hidden_sizes=pi_hidden_sizes
        )

        self.q_nn_target.set_params_to(self.q_nn)
        self.pi_nn_target.set_params_to(self.pi_nn)
        self.q_nn_target.disable_grad()
        self.pi_nn_target.disable_grad()

        self.replay_buffer = core.Replay_buffer(size=replay_buffer_size)
        self.nns = [self.q_nn, self.pi_nn, self.q_nn_target, self.pi_nn_target]

    def action(self):
        return self.pi_nn.action(self.state, noise=self.action_noise)

    def train(self, epochs, epoch_size, batch_size, max_episode_length):
        """Train the algorithm

        :param max_episode_length: number of steps in which to initially take
         random actions
        :type max_episode_length: int
        """

        self.init_experiment()

        for epoch in range(epochs):
            self.init_epoch(epoch)
            epoch_q_values = []
            epoch_q_loss = []
            epoch_pi_loss = []

            while self.t_epoch < epoch_size:
                self.init_episode()

                while not self.episode_done:
                    # End the episode if it's too long
                    if self.terminate_episode_early(max_episode_length, epoch_size):
                        break
                    self.episode_step()

                # TODO Should we be adding episodes to the buffer if they are
                # not finished?
                self.replay_buffer.update(self.episode)

                # Do an update for each step in the episode
                # XXX Seems like a strange way to set the frequency of updates
                for _ in range(self.t_episode):

                    batch = self.replay_buffer.sample(size=batch_size)

                    # Update the q network
                    self.zero_grad()
                    self.q_nn.ddpg_update(
                        batch=batch,
                        target_pi_network=self.pi_nn_target,
                        target_q_network=self.q_nn_target,
                    )

                    epoch_q_values.append(self.q_nn.recent_mean_q_values.numpy())
                    epoch_q_loss.append(self.q_nn.recent_loss.numpy())

                    # Update the policy network
                    self.zero_grad()
                    theo_actions = self.pi_nn.forward(batch["state"])
                    theo_state_actions = torch.cat(
                        [batch["state"], theo_actions], dim=1
                    )
                    pi_loss = -self.q_nn.forward(theo_state_actions)
                    pi_loss = torch.mean(pi_loss)

                    pi_loss.backward()
                    self.pi_nn.optimiser.step()
                    epoch_pi_loss.append(pi_loss.detach().numpy())

                    # Polyak update the target networks
                    self.pi_nn_target.polyak_update(self.pi_nn, self.rho)
                    self.q_nn_target.polyak_update(self.q_nn, self.rho)

                self.end_episode()

            self.writer.add_scalar("mean-pi-loss", np.mean(epoch_pi_loss), epoch)
            self.writer.add_scalar("mean-q-loss", np.mean(epoch_q_loss), epoch)
            self.writer.add_scalar("mean-q-values", np.mean(epoch_q_values), epoch)
            self.end_epoch(epoch)
            # Next epoch

        # End of epochs
        print("Finished training")
