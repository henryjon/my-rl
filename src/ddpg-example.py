import gym

import ddpg

test = False

ddpg_alg = ddpg.DDPG(
    env_fn=lambda: gym.make("HalfCheetah-v2"),
    hidden_sizes=(64,),
    gamma=0.99,
    rho=0.995,
    action_noise=0.1,
    replay_buffer_size=3 if test else 1_000_000,
    log_dir="/Users/harrygiles/tmp/my-rl/ddpg",
)

ddpg_alg.train(
    epochs=2 if test else 50,
    epoch_size=10 if test else 5_000,
    batch_size=10 if test else 100,
    exploration_period=10_000,
    max_episode_length=1_000,
)
