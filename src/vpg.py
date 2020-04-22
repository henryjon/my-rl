import core


class VPG(core.Algorithm):
    """A vanilla policy gradient algorithm

    Finite horizon un-discounted reward. Initialised with a policy network
    """

    def __init__(self, pi_hidden_sizes, *args, alpha=1e-3, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.pi_nn = core.Discrete_policy_network(
            self.env, hidden_sizes=pi_hidden_sizes
        )
        self.nns = [self.pi_nn]

    def action(self):
        return self.pi_nn.action(self.state)

    def train(self, epoch_size, epochs, max_episode_length):
        """Trains the algorithm"""

        self.init_experiment()

        for epoch in range(epochs):
            self.init_epoch(epoch)
            epoch_batch = []

            while self.t_epoch < epoch_size:

                self.init_episode()

                while not self.episode_done:
                    # End the episode if it's too long
                    if self.terminate_episode_early(max_episode_length, epoch_size):
                        break

                    self.episode_step()

                epoch_batch.append(self.episode)
                self.end_episode()

            self.pi_nn.policy_gradient_update(
                epoch_batch, writer=self.writer, epoch=epoch
            )

            self.end_epoch(epoch)
            # Next epoch

        # End of epochs
        print("Finished training")
