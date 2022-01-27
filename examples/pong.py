import gym
import dreamerv2.lib as dv2

config = dv2.defaults.update({
    'logdir': '~/logdir/pong',
    'resize': [64, 64],
    'grayscale': True,
    'log_every': 1e3,
    'prefill': 1e5,
    'save_every': 1e5
}).parse_flags()
import gym
env = gym.make('Pong-v0')
dreamerv2 = dv2.DreamerV2(env, config)
dreamerv2.train()