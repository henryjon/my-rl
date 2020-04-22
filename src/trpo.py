import core


class TRPO(core.Algorithm):
    """A Trust Region Policy Optimisation algorithm for discrete action space

    :param v_hidden_sizes: the hidden layer sizes of the value network

    :param pi_hidden_sizes: the hidden layer sizes of the policy network

    :param max_backtracking_steps: for backtracking

    :param backtracking_rate: for backtracking

    :param kl_tolerance: between updates
    """

    def __init__(
        self,
        pi_hidden_sizes,
        v_hidden_sizes,
        max_backtracking_steps,
        backtracking_rate,
        kl_tolerance,
        gae_lambda,
        *args,
        train_v_ratio=80,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.pi_nn = core.Discrete_policy_network(
            self.env, hidden_sizes=pi_hidden_sizes
        )
        self.v_nn = core.Discrete_value_network(self.env, hidden_sizes=v_hidden_sizes)
        self.nns = [self.pi_nn, self.v_nn]

        self.max_backtracking_steps = max_backtracking_steps
        self.backtracking_rate = backtracking_rate
        self.kl_tolerance = kl_tolerance
        self.gae_lambda = gae_lambda
        self.train_v_ratio = train_v_ratio

    def action(self):
        return self.pi_nn.action(self.state)

    def train(self, epochs, epoch_size, max_episode_length):
        """Train the algorithm

        :param max_episode_length: number of steps in which to initially take
         random actions
        :type max_episode_length: int
        """

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

            # Update the policy network

            g = self.pi_nn.compute_policy_gradient(
                batch=epoch_batch,
                return_type="gradient",
                weights_type="advantage_estimates",
                v_fn=self.v_nn.forward,
                gae_lambda=self.gae_lambda,
            )
            H_inv_g = core.lin_solve(
                apply_A_fn=self.pi_nn.KL_hessian_matmul_fn(batch=epoch_batch), b=g
            )

            k = 2 * self.kl_tolerance
            k /= sum((t1 * t2).sum() for t1, t2 in zip(H_inv_g, g))
            k = k ** 0.5

            improved = False
            for _ in range(self.max_backtracking_steps):
                k *= self.backtracking_rate
                pi_new = core.Discrete_policy_network(
                    self.env, hidden_sizes=self.pi_nn.sizes[1:-1]
                )
                pi_new.set_params_to(self.pi_nn)
                pi_new.add_to_parameters([k * t for t in H_inv_g])
                KL = self.pi_nn.KL(batch=epoch_batch, pi_other=pi_new)
                SA = self.pi_nn.surrogate_advantage_approxim(pi_other=pi_new, g=g)
                if KL < self.kl_tolerance and SA > 0:
                    improved = True
                    break

            self.pi_nn.set_params_to(pi_new)
            del pi_new

            self.writer.add_scalar("kl", KL, epoch)
            self.writer.add_scalar("sa", SA, epoch)
            self.writer.add_scalar("improved", improved, epoch)

            # Update the value network
            for _ in range(self.train_v_ratio):
                self.v_nn.simple_update(
                    batch=epoch_batch, writer=self.writer, n_iter=epoch
                )

            self.end_epoch(epoch)
            # Next epoch

        # End of epochs
        print("Finished training")
