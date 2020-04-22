import gym

import vpg

test = False

vpg_alg = vpg.VPG(
    env_fn=lambda: gym.make("CartPole-v1"),
    pi_hidden_sizes=(64,),
    log_dir="/Users/harrygiles/tmp/my-rl/vpg",
    exploration_period=None,
)


vpg_alg.train(
    epoch_size=100 if test else 4_000,
    epochs=2 if test else 50,
    max_episode_length=1_000,
)
