import gym

import trpo

test = False

trpo_alg = trpo.TRPO(
    env_fn=lambda: gym.make("CartPole-v1"),
    pi_hidden_sizes=(2,) if test else (64,),
    v_hidden_sizes=(2,) if test else (64,),
    backtracking_rate=0.8,
    max_backtracking_steps=10,
    kl_tolerance=0.01,
    gae_lambda=0.97,
    log_dir="/Users/harrygiles/tmp/my-rl/trpo",
    exploration_period=None,
)


trpo_alg.train(
    epoch_size=100 if test else 4_000,
    epochs=2 if test else 50,
    max_episode_length=1_000,
)
