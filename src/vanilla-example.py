import gym

import vanilla

test = False

vanilla_alg = vanilla.Vanilla(
    env_fn=lambda: gym.make("CartPole-v1"),
    hidden_sizes=(64,),
    log_dir="/Users/harrygiles/tmp/my-rl/vanilla",
)


vanilla_alg.train(
    epoch_size=100 if test else 4_000,
    epochs=2 if test else 50,
    max_episode_length=1_000,
)
